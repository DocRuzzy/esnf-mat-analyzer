import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import time # Added for save_debug_image

from ..core.interfaces import ImageProcessorInterface
from ..core.data_types import ProcessingConfig, GrayscaleConversionMethod

class ImageProcessor(ImageProcessorInterface):
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
        Load an image from the specified path. Converts to RGB.

        Args:
            path: Path to the image file.

        Returns:
            Loaded image as a numpy array (RGB).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be loaded or is invalid.
        """
        if not path.exists():
            self.logger.error(f"Image file not found at {path}")
            raise FileNotFoundError(f"Image file not found at {path}")

        try:
            image = cv2.imread(str(path))
            if image is None:
                self.logger.error(f"Failed to load image from {path} (cv2.imread returned None).")
                raise ValueError(f"Failed to load image from {path}. File might be corrupted or an unsupported format.")

            # Convert BGR (OpenCV default) to RGB for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Image loaded from {path}, shape: {image_rgb.shape}")
            return image_rgb
        except Exception as e:
            self.logger.error(f"Error loading image {path}: {e}", exc_info=True)
            raise ValueError(f"Error loading image {path}: {e}")


    def _apply_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion based on config."""
        if len(image.shape) == 2: # Already grayscale
            return image
        if image.shape[2] != 3: # Not a 3-channel color image
             self.logger.warning("Cannot apply grayscale to non-3-channel image. Returning as is.")
             return image

        method = self.config.grayscale_conversion
        if method == GrayscaleConversionMethod.WEIGHTED:
            # Standard RGB to Grayscale conversion: Y = 0.299R + 0.587G + 0.114B
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif method == GrayscaleConversionMethod.AVERAGE:
            gray_image = np.mean(image, axis=2).astype(np.uint8)
        elif method == GrayscaleConversionMethod.LUMINANCE: # Perceptual luminance (closer to human perception)
            # Using a common formula, slightly different from OpenCV's default weighted
            gray_image = (0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]).astype(np.uint8)
        else:
            self.logger.warning(f"Unknown grayscale method: {method}. Defaulting to WEIGHTED.")
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.logger.debug(f"Applied grayscale conversion using {method}.")
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
        """Applies background leveling using a median blur."""
        if not self.config.leveling.enabled:
            return image

        self.logger.debug("Applying background leveling.")
        kernel_size = self.config.leveling.kernel_size

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Estimate background with a median blur
        background = cv2.medianBlur(image, kernel_size)

        # Subtract background
        leveled_image = cv2.subtract(image, background)

        # Invert back
        leveled_image = cv2.bitwise_not(leveled_image)

        self.logger.debug("Background leveling complete.")
        return leveled_image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for analysis.
        Applies grayscale, blur, and contrast adjustments based on config.
        Input image is expected to be RGB.
        Output image is grayscale.
        """
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

    def crop_image(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops an image to a given region of interest.

        Args:
            image: The image to crop.
            roi: A tuple (x1, y1, x2, y2) representing the bounding box.

        Returns:
            The cropped image.
        """
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2]

    def analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyzes and returns basic properties of the image."""
        properties = {}
        properties["shape"] = image.shape
        properties["dtype"] = str(image.dtype)

        if len(image.shape) == 3:
            properties["channels"] = image.shape[2]
        elif len(image.shape) == 2:
            properties["channels"] = 1
        else:
            properties["channels"] = "Unknown"

        properties["min_value"] = int(np.min(image))
        properties["max_value"] = int(np.max(image))
        properties["mean_value"] = float(np.mean(image))
        properties["std_deviation"] = float(np.std(image))

        self.logger.debug(f"Analyzed image properties: {properties}")
        return properties

    def save_debug_image(self, image: np.ndarray, output_dir: Path, prefix: str) -> Path:
        """Saves an image to the debug directory for inspection."""
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = output_dir / filename

        try:
            # If image is RGB, convert to BGR for OpenCV imwrite
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else: # Grayscale or other, save as is
                image_to_save = image
            cv2.imwrite(str(filepath), image_to_save)
            self.logger.info(f"Saved debug image to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save debug image {filepath}: {e}", exc_info=True)
            raise
