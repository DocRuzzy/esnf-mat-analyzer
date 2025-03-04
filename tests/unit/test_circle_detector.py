"""
Tests for the CircleDetector class.

This module contains unit tests for the CircleDetector class to ensure
it correctly detects circular regions in nanofiber mat images.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import cv2

from esnf_mat_analyzer.core.data_types import (
    CircleDetectionConfig,
    CircleDetectionMethod,
)
from esnf_mat_analyzer.processing.circle_detector import CircleDetector


class TestCircleDetector:
    """Test suite for the CircleDetector class."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with a white circle on black background."""
        # Create a black image
        image = np.zeros((300, 300), dtype=np.uint8)

        # Draw white circle
        center = (150, 150)
        radius = 80
        cv2.circle(image, center, radius, 255, -1)

        return image, center, radius

    @pytest.fixture
    def noisy_image(self):
        """Create a noisy test image with a white circle on black background."""
        # Create a black image
        image = np.zeros((300, 300), dtype=np.uint8)

        # Draw white circle
        center = (150, 150)
        radius = 80
        cv2.circle(image, center, radius, 255, -1)

        # Add noise
        noise = np.random.normal(0, 25, image.shape).astype(np.int16)
        noisy_img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return noisy_img, center, radius

    @pytest.fixture
    def eccentric_image(self):
        """Create a test image with an off-center white circle."""
        # Create a black image
        image = np.zeros((300, 300), dtype=np.uint8)

        # Draw white circle near edge
        center = (50, 250)
        radius = 40
        cv2.circle(image, center, radius, 255, -1)

        return image, center, radius

    @pytest.fixture
    def default_config(self):
        """Return a default circle detection configuration."""
        return CircleDetectionConfig()

    def test_detect_hough(self, test_image, default_config):
        """Test circle detection using Hough transform."""
        # Set detection method to Hough
        config = CircleDetectionConfig(
            detection_method=CircleDetectionMethod.HOUGH, min_radius=50, max_radius=100
        )
        detector = CircleDetector(config)

        image, true_center, true_radius = test_image
        detected_center, detected_radius = detector.detect(image)

        # Check that the detected center and radius are close to the actual values
        assert abs(detected_center[0] - true_center[0]) <= 5
        assert abs(detected_center[1] - true_center[1]) <= 5
        assert abs(detected_radius - true_radius) <= 5

    def test_detect_contour(self, test_image, default_config):
        """Test circle detection using contour-based method."""
        # Set detection method to contour
        config = CircleDetectionConfig(
            detection_method=CircleDetectionMethod.CONTOUR,
            min_radius=50,
            max_radius=100,
            min_area=10000,
            max_area=100000,
        )
        detector = CircleDetector(config)

        image, true_center, true_radius = test_image
        detected_center, detected_radius = detector.detect(image)

        # Check that the detected center and radius are close to the actual values
        assert abs(detected_center[0] - true_center[0]) <= 5
        assert abs(detected_center[1] - true_center[1]) <= 5
        assert abs(detected_radius - true_radius) <= 5

    def test_detect_with_noise(self, noisy_image, default_config):
        """Test circle detection with noisy image."""
        detector = CircleDetector(default_config)

        image, true_center, true_radius = noisy_image
        detected_center, detected_radius = detector.detect(image)

        # Check that the detected center and radius are reasonably close despite noise
        assert abs(detected_center[0] - true_center[0]) <= 10
        assert abs(detected_center[1] - true_center[1]) <= 10
        assert abs(detected_radius - true_radius) <= 10

    def test_detect_eccentric(self, eccentric_image, default_config):
        """Test detection of an off-center circle."""
        detector = CircleDetector(default_config)

        image, true_center, true_radius = eccentric_image
        detected_center, detected_radius = detector.detect(image)

        # Check that the detected center and radius are close to the actual values
        assert abs(detected_center[0] - true_center[0]) <= 10
        assert abs(detected_center[1] - true_center[1]) <= 10
        assert abs(detected_radius - true_radius) <= 10

    def test_create_mask(self, default_config):
        """Test creating a circular mask."""
        detector = CircleDetector(default_config)

        # Create a mask for a 200x200 image with a circle at center (100,100) and radius 50
        image_shape = (200, 200)
        center = (100, 100)
        radius = 50

        mask = detector.create_mask(image_shape, center, radius)

        # Check that the mask has the correct shape and type
        assert mask.shape == image_shape
        assert mask.dtype == np.uint8

        # Check that the center is masked in
        assert mask[center[1], center[0]] == 1

        # Check that corners are masked out
        assert mask[0, 0] == 0
        assert mask[0, 199] == 0
        assert mask[199, 0] == 0
        assert mask[199, 199] == 0

        # Check that points within the radius are masked in
        assert mask[100, 75] == 1  # 25 pixels left of center
        assert mask[75, 100] == 1  # 25 pixels above center

        # Check that points outside the radius are masked out
        assert mask[100, 25] == 0  # 75 pixels left of center
        assert mask[25, 100] == 0  # 75 pixels above center

        # Check that the total area of the mask is close to πr²
        mask_area = np.sum(mask)
        expected_area = np.pi * radius**2
        assert abs(mask_area - expected_area) < expected_area * 0.1  # Within 10%

    def test_fallback_detection(self, default_config):
        """Test fallback detection method when primary methods fail."""
        detector = CircleDetector(default_config)

        # Create an image with no clear circle (low contrast)
        image = np.ones((200, 200), dtype=np.uint8) * 127

        # Add slight brightness in the center to guide the fallback method
        y, x = np.ogrid[:200, :200]
        r = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
        image[r < 50] = 150

        # Set extreme parameters to force fallback method
        extreme_config = CircleDetectionConfig(
            min_radius=150, max_radius=200  # Larger than any circle in the image
        )
        detector = CircleDetector(extreme_config)

        center, radius = detector.detect(image)

        # Check that fallback method produced reasonable values
        assert 75 <= center[0] <= 125
        assert 75 <= center[1] <= 125
        assert radius >= extreme_config.min_radius

    def test_visualize_detection(self, test_image, default_config):
        """Test visualization of circle detection."""
        detector = CircleDetector(default_config)

        image, center, radius = test_image

        # Visualize detection
        vis_image = detector.visualize_detection(image, center, radius)

        # Check that the visualization has the correct shape and type
        assert vis_image.shape == (image.shape[0], image.shape[1], 3)
        assert vis_image.dtype == np.uint8

        # Convert image to RGB if it's not already
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image

        # Check that the visualization is different from the original
        # (should have added graphics)
        assert not np.array_equal(vis_image, rgb_image)

    def test_evaluate_detection_quality(self, test_image, default_config):
        """Test evaluation of detection quality."""
        detector = CircleDetector(default_config)

        image, center, radius = test_image

        # Evaluate detection quality
        quality = detector.evaluate_detection_quality(image, center, radius)

        # Check that the quality dictionary contains expected keys
        expected_keys = [
            "contrast_ratio",
            "inside_std",
            "edge_strength",
            "symmetry",
            "overall_quality",
        ]
        for key in expected_keys:
            assert key in quality

        # Perfect circle on black background should have high quality metrics
        assert quality["contrast_ratio"] > 1.0  # White inside, black outside
        assert quality["edge_strength"] > 0.0
        assert quality["symmetry"] > 0.8  # High symmetry expected
        assert quality["overall_quality"] > 0.5  # Overall quality should be good
