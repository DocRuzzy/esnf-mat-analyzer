import cv2
import numpy as np
import logging
from typing import Optional

from esnf_mat_analyzer.core.interfaces import ThicknessEstimatorInterface
from esnf_mat_analyzer.core.data_types import ThicknessConfig, ThicknessModelType
from esnf_mat_analyzer.processing.physics_based_thickness import BeerLambertEstimator

class ThicknessEstimator(ThicknessEstimatorInterface):
    """
    Estimates thickness from image brightness.
    """

    def __init__(self, config: ThicknessConfig):
        """
        Initialize the ThicknessEstimator.

        Args:
            config: Configuration for thickness estimation.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ThicknessEstimator initialized with config: {self.config}")
        
        # Initialize Beer-Lambert estimator if needed
        if config.model_type == ThicknessModelType.BEER_LAMBERT:
            self.beer_lambert = BeerLambertEstimator(
                reflection_coefficient=config.attenuation_coefficient,
                background_intensity=None,  # Auto-detect
                max_reflection=None,  # Auto-detect
                max_thickness_um=config.thickness_range_um[1]
            )

    def estimate(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Estimate thickness from image brightness.

        Args:
            image: Preprocessed grayscale image (assumed to be 0-255 range).
            mask: Binary mask indicating the region of interest.

        Returns:
            2D thickness map. Units depend on model and calibration_factor.
        """
        self.logger.debug(f"Estimating thickness for image of shape {image.shape} using mask.")

        if len(image.shape) != 2:
            self.logger.error("Input image for thickness estimation must be grayscale.")
            raise ValueError("Input image for thickness estimation must be grayscale.")

        # Apply the chosen thickness model
        # Ensure calculations are done in float to maintain precision
        image_float = image.astype(np.float32)

        if self.config.model_type == ThicknessModelType.BEER_LAMBERT:
            # Use Beer-Lambert law for physics-based thickness estimation
            self.logger.info("Using Beer-Lambert law for thickness estimation")
            thickness_map = self.beer_lambert.estimate_thickness(
                image,
                background_corrected=True,
                spatial_scale_um_per_pixel=self.config.spatial_scale_um_per_pixel
            )
        elif self.config.model_type == ThicknessModelType.LINEAR:
            # thickness = a * brightness + b
            thickness_map = self.config.a * image_float + self.config.b
        elif self.config.model_type == ThicknessModelType.LOGARITHMIC:
            # thickness = a * log(1 + brightness) + b
            # Adding 1 to brightness to avoid log(0)
            thickness_map = self.config.a * np.log1p(image_float) + self.config.b
        elif self.config.model_type == ThicknessModelType.EXPONENTIAL:
            # thickness = a * (exp(brightness / 255) - 1) + b
            normalized_brightness = image_float / 255.0
            thickness_map = self.config.a * (np.exp(normalized_brightness) - 1) + self.config.b
        else:
            self.logger.error(f"Unknown thickness model type: {self.config.model_type}")
            raise ValueError(f"Unknown thickness model type: {self.config.model_type}")

        # Apply mask - set thickness to 0 (or NaN) outside the ROI
        thickness_map[mask == 0] = 0.0 # Or np.nan if preferred downstream

        # Apply calibration factor if provided
        if self.config.calibration_factor is not None and self.config.calibration_factor > 0:
            self.logger.info(f"Applying calibration factor: {self.config.calibration_factor}")
            thickness_map *= self.config.calibration_factor
        else:
            self.logger.info("No calibration factor applied or factor is invalid.")

        # Normalization (optional, based on config)
        if self.config.normalization:
            self.logger.info("Normalizing thickness map to [0, 1] range.")
            min_val = np.min(thickness_map[mask > 0]) if np.any(thickness_map[mask > 0]) else 0
            max_val = np.max(thickness_map[mask > 0]) if np.any(thickness_map[mask > 0]) else 0
            if max_val > min_val:
                thickness_map = (thickness_map - min_val) / (max_val - min_val)
                thickness_map[mask == 0] = 0.0 # Restore masked area if changed by normalization
            elif max_val == min_val and max_val > 0: # All values in mask are the same and positive
                 thickness_map[mask > 0] = 1.0
            # else: all zeros or single value, no normalization needed or possible

        # Ensure non-negative thickness
        thickness_map[thickness_map < 0] = 0.0

        self.logger.debug("Thickness estimation complete.")
        return thickness_map.astype(np.float32) # Ensure float output

    def get_saturation_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a mask of saturated pixels.
        Pixels are considered saturated if their brightness value is at or above
        the configured saturation_threshold.

        Args:
            image: Preprocessed grayscale image.
            mask: Binary mask indicating the region of interest.

        Returns:
            Binary mask where 1 indicates saturated pixels within the ROI.
        """
        self.logger.debug(f"Generating saturation mask with threshold: {self.config.saturation_threshold}")
        if len(image.shape) != 2:
            self.logger.error("Input image for saturation mask must be grayscale.")
            raise ValueError("Input image for saturation mask must be grayscale.")

        saturation_mask_full = (image >= self.config.saturation_threshold).astype(np.uint8)

        # Only consider saturation within the provided ROI mask
        saturation_mask_roi = saturation_mask_full * mask

        num_saturated_pixels = np.sum(saturation_mask_roi)
        self.logger.info(f"Found {num_saturated_pixels} saturated pixels within the ROI.")

        return saturation_mask_roi

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Dummy config
    config = ThicknessConfig(
        model_type=ThicknessModelType.LINEAR,
        a=1.0,
        b=0.0,
        saturation_threshold=250,
        calibration_factor=10.0 # e.g., 10 nm per unit brightness
    )
    estimator = ThicknessEstimator(config)

    # Dummy image and mask
    dummy_height, dummy_width = 100, 100
    dummy_image = np.random.randint(0, 256, (dummy_height, dummy_width), dtype=np.uint8)
    dummy_image[10:20, 10:20] = 255 # Saturated region

    dummy_mask = np.zeros((dummy_height, dummy_width), dtype=np.uint8)
    cv2.circle(dummy_mask, (dummy_width//2, dummy_height//2), dummy_height//3, 1, -1)


    logger.info("Estimating thickness on dummy image...")
    thickness_map_result = estimator.estimate(dummy_image, dummy_mask)
    logger.info(f"Thickness map generated. Shape: {thickness_map_result.shape}, Data type: {thickness_map_result.dtype}")
    logger.info(f"Sample thickness values (center): {thickness_map_result[dummy_height//2, dummy_width//2]}")


    logger.info("Getting saturation mask...")
    saturation_mask_result = estimator.get_saturation_mask(dummy_image, dummy_mask)
    logger.info(f"Saturation mask generated. Shape: {saturation_mask_result.shape}, Num saturated: {np.sum(saturation_mask_result)}")

    # Optional: Display results if cv2 has GUI capabilities
    # combined_display = np.hstack([
    #     cv2.cvtColor(dummy_image, cv2.COLOR_GRAY2BGR),
    #     cv2.cvtColor(thickness_map_result.astype(np.uint8), cv2.COLOR_GRAY2BGR), # Rough visualization
    #     cv2.cvtColor(saturation_mask_result * 255, cv2.COLOR_GRAY2BGR)
    # ])
    # cv2.imshow("Original | Thickness (Vis) | Saturation", combined_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
