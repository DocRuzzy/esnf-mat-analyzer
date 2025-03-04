"""
Circle detector for the ESNF Mat Analyzer.

This module provides functionality for detecting circular regions in
electrospun nanofiber (ESNF) mat images. It implements the CircleDetectorInterface
defined in the core module.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union, cast

from esnf_mat_analyzer.core.interfaces import CircleDetectorInterface
from esnf_mat_analyzer.core.data_types import (
    CircleDetectionConfig,
    CircleDetectionMethod,
    Point,
)


class CircleDetector(CircleDetectorInterface):
    """
    Component for detecting circular regions in nanofiber mat images.

    This class implements the CircleDetectorInterface and provides
    functionality for detecting circular regions using different methods
    and creating binary masks for the detected regions.
    """

    def __init__(self, config: CircleDetectionConfig):
        """
        Initialize the circle detector with configuration settings.

        Args:
            config: Configuration parameters for circle detection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray) -> Tuple[Point, int]:
        """
        Detect the circular region in the image.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Tuple of ((center_x, center_y), radius)

        Raises:
            ValueError: If no circular region can be detected
        """
        self.logger.debug("Detecting circular region")

        # Ensure image is grayscale
        if len(image.shape) > 2:
            self.logger.warning("Input image is not grayscale, converting")
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            gray = image

        # Perform circle detection based on configured method
        if self.config.detection_method == CircleDetectionMethod.HOUGH:
            center, radius = self._detect_hough(gray)
        elif self.config.detection_method == CircleDetectionMethod.CONTOUR:
            center, radius = self._detect_contour(gray)
        else:
            raise ValueError(
                f"Unknown detection method: {self.config.detection_method}"
            )

        # Validate detection results
        if not self._validate_circle(center, radius, gray.shape):
            self.logger.warning("Invalid circle detected, using fallback")
            center, radius = self._fallback_detection(gray)

        self.logger.info(f"Detected circle at {center} with radius {radius}")
        return center, radius

    def _detect_hough(self, image: np.ndarray) -> Tuple[Point, int]:
        """
        Detect circles using the Hough Circle Transform.

        Args:
            image: Grayscale image

        Returns:
            Tuple of ((center_x, center_y), radius)
        """
        self.logger.debug("Using Hough Circle Transform for detection")

        # Apply median blur to reduce noise and improve circle detection
        blurred = cv2.medianBlur(image, 5)

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,  # Resolution ratio of accumulator to image
            minDist=image.shape[0] // 2,  # Minimum distance between circles
            param1=self.config.param1,  # Upper threshold for edge detection
            param2=self.config.param2,  # Threshold for circle detection
            minRadius=self.config.min_radius,
            maxRadius=self.config.max_radius,
        )

        # Check if any circles were detected
        if circles is not None and len(circles[0]) > 0:
            circles = np.uint16(np.around(circles))

            # Sort circles by prominence (param2 corresponds to number of points for the circle)
            # We approximate this by selecting circles with radius closer to the median
            if len(circles[0]) > 1:
                self.logger.debug(
                    f"Multiple circles detected ({len(circles[0])}), selecting best match"
                )
                radii = [c[2] for c in circles[0]]
                median_radius = np.median(radii)
                best_idx = np.argmin([abs(r - median_radius) for r in radii])
                best_circle = circles[0][best_idx]
            else:
                best_circle = circles[0][0]

            center = (int(best_circle[0]), int(best_circle[1]))
            radius = int(best_circle[2])

            self.logger.debug(f"Hough detected circle at {center} with radius {radius}")
            return center, radius
        else:
            self.logger.warning("No circles detected with Hough transform")
            return self._fallback_detection(image)

    def _detect_contour(self, image: np.ndarray) -> Tuple[Point, int]:
        """
        Detect circles using contour-based methods.

        Args:
            image: Grayscale image

        Returns:
            Tuple of ((center_x, center_y), radius)
        """
        self.logger.debug("Using contour-based method for detection")

        # Apply threshold to create binary image
        # Use Otsu's method for automatic thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_area <= area <= self.config.max_area:
                valid_contours.append(contour)

        if not valid_contours:
            self.logger.warning("No valid contours found")
            return self._fallback_detection(image)

        # Find the largest contour (presumably the nanofiber mat)
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Fit a minimum enclosing circle
        center, radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        # Check if the radius is within acceptable range
        if not (self.config.min_radius <= radius <= self.config.max_radius):
            self.logger.warning(
                f"Detected radius ({radius}) is outside acceptable range"
            )
            return self._fallback_detection(image)

        self.logger.debug(f"Contour detected circle at {center} with radius {radius}")
        return center, radius

    def _fallback_detection(self, image: np.ndarray) -> Tuple[Point, int]:
        """
        Fallback method for circle detection when primary methods fail.

        This method uses image moments and intensity-based analysis
        to estimate the center and radius.

        Args:
            image: Grayscale image

        Returns:
            Tuple of ((center_x, center_y), radius)
        """
        self.logger.debug("Using fallback detection method")

        # Try to use image moments to find center of mass
        height, width = image.shape
        y_indices, x_indices = np.where(image > np.mean(image))

        if len(y_indices) > 0 and len(x_indices) > 0:
            # Calculate center as mean of bright pixels
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))

            # Calculate radius as half the mean distance from center to bright pixels
            distances = np.sqrt(
                (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
            )
            radius = int(np.mean(distances))

            # Ensure radius is within acceptable range
            radius = max(self.config.min_radius, min(radius, self.config.max_radius))

            center = (center_x, center_y)
        else:
            # Default to image center and half of minimum dimension
            center = (width // 2, height // 2)
            radius = min(width, height) // 4
            radius = max(self.config.min_radius, min(radius, self.config.max_radius))

            self.logger.warning("Using default center and radius as last resort")

        self.logger.debug(f"Fallback detected circle at {center} with radius {radius}")
        return center, radius

    def _validate_circle(
        self, center: Point, radius: int, image_shape: Tuple[int, int]
    ) -> bool:
        """
        Validate that the detected circle is reasonable.

        Args:
            center: Circle center coordinates (x, y)
            radius: Circle radius
            image_shape: Image dimensions (height, width)

        Returns:
            True if the circle is valid, False otherwise
        """
        height, width = image_shape
        x, y = center

        # Check that radius is within configured range
        if not (self.config.min_radius <= radius <= self.config.max_radius):
            self.logger.warning(
                f"Radius {radius} outside configured range "
                f"({self.config.min_radius}-{self.config.max_radius})"
            )
            return False

        # Check that circle is mostly within the image
        if x < 0 or x >= width or y < 0 or y >= height:
            self.logger.warning(f"Center {center} outside image boundaries")
            return False

        # Check that at least 50% of the circle is within the image
        left_bound = max(0, x - radius)
        right_bound = min(width - 1, x + radius)
        top_bound = max(0, y - radius)
        bottom_bound = min(height - 1, y + radius)

        visible_width = right_bound - left_bound
        visible_height = bottom_bound - top_bound

        if visible_width < radius or visible_height < radius:
            self.logger.warning("Less than 50% of circle is within image boundaries")
            return False

        return True

    def create_mask(
        self, image_shape: Tuple[int, int], center: Point, radius: int
    ) -> np.ndarray:
        """
        Create a binary mask for the circular region.

        Args:
            image_shape: Shape of the image (height, width)
            center: Center coordinates (x, y)
            radius: Radius of the circle

        Returns:
            Binary mask where 1 indicates the circular region
        """
        self.logger.debug(f"Creating circular mask at {center} with radius {radius}")

        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled circle on the mask
        cv2.circle(mask, center, radius, 1, thickness=-1)

        return mask

    def visualize_detection(
        self, image: np.ndarray, center: Point, radius: int
    ) -> np.ndarray:
        """
        Create a visualization of the detected circle.

        Args:
            image: Original image
            center: Center coordinates (x, y)
            radius: Radius of the circle

        Returns:
            Image with circle overlay
        """
        # Create a copy of the image to avoid modifying the original
        vis_image = image.copy()

        # Convert to color if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)

        # Draw circle outline
        cv2.circle(vis_image, center, radius, (0, 255, 0), 2)

        # Draw center point
        cv2.circle(vis_image, center, 3, (255, 0, 0), -1)

        # Draw radius line
        end_point = (center[0] + int(radius * 0.8), center[1])
        cv2.line(vis_image, center, end_point, (255, 0, 0), 1)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Label for center coordinates
        center_text = f"Center: ({center[0]}, {center[1]})"
        cv2.putText(
            vis_image,
            center_text,
            (center[0] + 10, center[1] + 20),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness + 1,
        )  # Add thicker outline for visibility
        cv2.putText(
            vis_image,
            center_text,
            (center[0] + 10, center[1] + 20),
            font,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

        # Label for radius
        radius_text = f"Radius: {radius}"
        cv2.putText(
            vis_image,
            radius_text,
            (center[0] + 10, center[1] + 40),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness + 1,
        )
        cv2.putText(
            vis_image,
            radius_text,
            (center[0] + 10, center[1] + 40),
            font,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

        return vis_image

    def evaluate_detection_quality(
        self, image: np.ndarray, center: Point, radius: int
    ) -> Dict[str, float]:
        """
        Evaluate the quality of circle detection.

        Args:
            image: Grayscale image
            center: Detected center coordinates (x, y)
            radius: Detected radius

        Returns:
            Dictionary with quality metrics
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            gray = image

        # Create circular mask
        mask = self.create_mask(gray.shape, center, radius)

        # Calculate average intensity inside and outside the circle
        inside_mean = np.mean(gray[mask > 0])
        outside_mean = np.mean(gray[mask == 0])

        # Calculate contrast ratio between inside and outside
        if outside_mean > 0:
            contrast_ratio = inside_mean / outside_mean
        else:
            contrast_ratio = float("inf")

        # Calculate standard deviation inside the circle
        inside_std = np.std(gray[mask > 0])

        # Calculate edge strength by finding average gradient magnitude at circle boundary
        edge_strength = self._calculate_edge_strength(gray, center, radius)

        # Calculate symmetry by comparing opposite sides of the circle
        symmetry = self._calculate_symmetry(gray, center, radius)

        return {
            "contrast_ratio": float(contrast_ratio),
            "inside_std": float(inside_std),
            "edge_strength": float(edge_strength),
            "symmetry": float(symmetry),
            "overall_quality": float((contrast_ratio + edge_strength + symmetry) / 3),
        }

    def _calculate_edge_strength(
        self, image: np.ndarray, center: Point, radius: int
    ) -> float:
        """
        Calculate the edge strength at the circle boundary.

        Args:
            image: Grayscale image
            center: Circle center coordinates (x, y)
            radius: Circle radius

        Returns:
            Average edge strength (0.0-1.0, higher is better)
        """
        # Generate points on the circle boundary
        num_points = 36  # Points every 10 degrees
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Calculate gradient magnitude using Sobel operator
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize magnitude to 0-1 range
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)

        # Sample gradient magnitude at circle boundary
        edge_values = []
        height, width = image.shape

        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))

            # Skip points outside image boundary
            if 0 <= x < width and 0 <= y < height:
                edge_values.append(magnitude[y, x])

        if not edge_values:
            return 0.0

        return float(np.mean(edge_values))

    def _calculate_symmetry(
        self, image: np.ndarray, center: Point, radius: int
    ) -> float:
        """
        Calculate the symmetry of the detected circle.

        Args:
            image: Grayscale image
            center: Circle center coordinates (x, y)
            radius: Circle radius

        Returns:
            Symmetry score (0.0-1.0, higher is more symmetric)
        """
        # Create a mask for the circle
        mask = self.create_mask(image.shape, center, radius)

        # Apply mask to get only the circle region
        masked_image = image.copy()
        masked_image[mask == 0] = 0

        # Calculate horizontal and vertical symmetry
        height, width = image.shape
        cx, cy = center

        # Calculate horizontal symmetry
        left_half = masked_image[:, :cx]
        right_half = masked_image[:, cx:]
        right_half_flipped = np.fliplr(right_half)

        # Adjust matrices to same size for comparison
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_compare = left_half[:, -min_width:]
        right_compare = right_half_flipped[:, :min_width]

        horizontal_diff = np.mean(
            np.abs(left_compare.astype(float) - right_compare.astype(float))
        )

        # Calculate vertical symmetry
        top_half = masked_image[:cy, :]
        bottom_half = masked_image[cy:, :]
        bottom_half_flipped = np.flipud(bottom_half)

        # Adjust matrices to same size for comparison
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_compare = top_half[-min_height:, :]
        bottom_compare = bottom_half_flipped[:min_height, :]

        vertical_diff = np.mean(
            np.abs(top_compare.astype(float) - bottom_compare.astype(float))
        )

        # Calculate average intensity for normalization
        avg_intensity = np.mean(masked_image[mask > 0])
        if avg_intensity == 0:
            return 0.0

        # Normalize differences and convert to symmetry score (higher is better)
        horizontal_symmetry = 1.0 - (horizontal_diff / avg_intensity)
        vertical_symmetry = 1.0 - (vertical_diff / avg_intensity)

        # Return average symmetry
        return float((horizontal_symmetry + vertical_symmetry) / 2.0)
