import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import time # Added for save_debug_image

from ..core.interfaces import IImageProcessor
from ..core.data_types import ProcessingConfig

class ImageProcessor(IImageProcessor):
    """
    Handles image loading and preprocessing operations.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the ImageProcessor.

        Args:
            config: Configuration for image processing.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ImageProcessor initialized with config: {self.config}")

    def load_image(self, path: Path) -> np.ndarray:
        """
        Load an image from the specified path.
        
        Args:
            path: Path to the image file
            
        Returns:
            Loaded image as a numpy array (RGB format)
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded
        """
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        try:
            # Load image using cv2 (loads in BGR format)
            image_bgr = cv2.imread(str(path))
            
            if image_bgr is None:
                raise ValueError(f"Could not load image from {path}. File may be corrupted or in an unsupported format.")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            self.logger.debug(f"Loaded image from {path} with shape {image_rgb.shape}")
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Error loading image from {path}: {e}")
            raise ValueError(f"Failed to load image: {e}")

    def process(self, image: np.ndarray, config: 'ProcessingConfig') -> np.ndarray:
        """Process raw image according to configuration."""
        self.logger.debug("Starting image preprocessing.")

        # 1. Grayscale conversion
        processed_image = self._apply_grayscale(image)

        # 2. Background Leveling
        processed_image = self._apply_leveling(processed_image)

        # 3. Blur
        processed_image = self._apply_blur(processed_image)

        # 4. Contrast/Brightness (on grayscale image)
        processed_image = self._apply_contrast_brightness(processed_image)

        self.logger.info("Image preprocessing complete.")
        return processed_image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess raw image using the instance configuration.
        
        Args:
            image: Raw image array (RGB format)
            
        Returns:
            Preprocessed grayscale image
        """
        return self.process(image, self.config)

    def crop_image(self, image: np.ndarray, roi: tuple) -> np.ndarray:
        """
        Crop image to the specified region of interest.
        
        Args:
            image: Input image array
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Cropped image array
            
        Raises:
            ValueError: If ROI is invalid or outside image bounds
        """
        if len(roi) != 4:
            raise ValueError(f"ROI must be a tuple of 4 values (x, y, width, height), got {len(roi)} values")
        
        x, y, width, height = roi
        
        # Validate ROI parameters
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid ROI parameters: x={x}, y={y}, width={width}, height={height}")
        
        # Check image bounds
        img_height, img_width = image.shape[:2]
        if x + width > img_width or y + height > img_height:
            self.logger.warning(f"ROI extends beyond image bounds. Image: {img_width}x{img_height}, ROI: {x},{y},{width},{height}")
            # Clamp to image bounds
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            self.logger.info(f"Adjusted ROI to fit image bounds: {x},{y},{width},{height}")
        
        # Crop the image
        cropped = image[y:y+height, x:x+width]
        
        self.logger.debug(f"Cropped image from {image.shape} to {cropped.shape} using ROI {roi}")
        return cropped

    def _apply_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion based on config."""
        if len(image.shape) == 2: # Already grayscale
            return image
        if image.shape[2] != 3: # Not a 3-channel color image
             self.logger.warning("Cannot apply grayscale to non-3-channel image. Returning as is.")
             return image

        method = self.config.grayscale_conversion
        
        # Handle both enum and string values for compatibility
        if hasattr(method, 'name'):
            method_name = method.name
        else:
            method_name = str(method).upper()
        
        if method_name == "WEIGHTED":
            # Standard RGB to Grayscale conversion: Y = 0.299R + 0.587G + 0.114B
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif method_name == "AVERAGE":
            gray_image = np.mean(image, axis=2).astype(np.uint8)
        elif method_name == "LUMINANCE": # Perceptual luminance (closer to human perception)
            # Using a common formula, slightly different from OpenCV's default weighted
            gray_image = (0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]).astype(np.uint8)
        else:
            self.logger.warning(f"Unknown grayscale method: {method}. Defaulting to WEIGHTED.")
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.logger.debug(f"Applied grayscale conversion using {method_name}.")
        return gray_image

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Applies Gaussian blur based on config."""
        kernel_size = self.config.blur_kernel_size
        if kernel_size > 0:
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            self.logger.debug(f"Applied Gaussian blur with kernel size {kernel_size}.")
            return blurred_image
        return image

    def _apply_contrast_brightness(self, image: np.ndarray) -> np.ndarray:
        """Applies contrast and brightness adjustment based on config."""
        alpha = self.config.contrast_alpha  # Contrast control (1.0-3.0)
        beta = self.config.contrast_beta    # Brightness control (0-100)

        # cv2.convertScaleAbs performs: output = saturate_cast(alpha * input + beta)
        # It handles saturation to 0-255 range.
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        self.logger.debug(f"Applied contrast (alpha={alpha}) and brightness (beta={beta}).")
        return adjusted_image

    def _apply_leveling(self, image: np.ndarray) -> np.ndarray:
        """
        Applies background correction to handle non-uniform illumination.
        
        This method preserves the natural relationship where:
        - White/bright areas (fiber accumulation) remain high values
        - Dark areas (background) become low values 
        - Non-uniform illumination is corrected based on the selected method
        """
        if not self.config.leveling.enabled:
            return image

        method = getattr(self.config, 'background_correction_method', None)
        
        if method is None or method.name == 'NONE':
            # Use the default normalization method for backward compatibility
            return self._apply_normalization_correction(image)
        
        from ..processing.advanced_background import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        
        if method.name == 'BASIC':
            return processor.basic_correction(image, self.config.basic_correction_n_components)
        elif method.name == 'ROLLING_BALL':
            return processor.rolling_ball_3d(image, self.config.rolling_ball_radius)
        elif method.name == 'RESTORE':
            return processor.restore_method(image, self.config.restore_percentile)
        elif method.name == 'HOMOMORPHIC':
            return processor.homomorphic_filter(image, self.config.homomorphic_cutoff, 
                                               self.config.homomorphic_g_low, self.config.homomorphic_g_high)
        else:
            self.logger.warning(f"Unknown background correction method: {method}, using default")
            return self._apply_normalization_correction(image)
            
    def _apply_normalization_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the default normalization-based background correction.
        """
        self.logger.debug("Applying normalization-based background leveling.")
        kernel_size = self.config.leveling.kernel_size

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Estimate background with a median blur
        background = cv2.medianBlur(image, kernel_size)
        
        # Convert to float to prevent overflow/underflow issues
        image_float = image.astype(np.float32)
        background_float = background.astype(np.float32)
        
        # Proper background correction: normalize by the background
        # This preserves bright areas as bright and corrects for illumination variation
        # Use a small offset to avoid division by zero
        epsilon = 1.0
        corrected = (image_float / (background_float + epsilon)) * 128.0
        
        # Clip to valid range and convert back to uint8
        leveled_image = np.clip(corrected, 0, 255).astype(np.uint8)

        self.logger.debug("Background leveling complete.")
        return leveled_image
