"""
Shape detector for the ESNF Mat Analyzer.

This module provides functionality for detecting general shapes in
electrospun nanofiber (ESNF) mat images.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List

class ShapeDetector:
    """
    Component for detecting the primary shape of the nanofiber mat.
    """

    def __init__(self):
        """
        Initialize the shape detector.
        """
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the primary contour of the mat in the image.

        Args:
            image: Preprocessed grayscale image

        Returns:
            The largest contour found in the image, or None if no contour is found.
        """
        self.logger.debug("Detecting shape contour")

        # Ensure image is grayscale
        if len(image.shape) > 2:
            self.logger.warning("Input image is not grayscale, converting")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply threshold to create binary image
        # Use Otsu's method for automatic thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self.logger.warning("No contours found")
            return None

        # Find the largest contour (presumably the nanofiber mat)
        largest_contour = max(contours, key=cv2.contourArea)

        self.logger.info(f"Detected contour with {len(largest_contour)} points.")
        return largest_contour

    def create_mask(self, image_shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for the detected shape.

        Args:
            image_shape: Shape of the image (height, width)
            contour: The contour of the shape.

        Returns:
            Binary mask where 1 indicates the shape region.
        """
        self.logger.debug("Creating shape mask")

        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled contour on the mask
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

        return mask

    def visualize_detection(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the detected shape.

        Args:
            image: Original image
            contour: The contour of the shape.

        Returns:
            Image with contour overlay.
        """
        # Create a copy of the image to avoid modifying the original
        vis_image = image.copy()

        # Convert to color if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)

        # Draw contour outline
        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)

        return vis_image
