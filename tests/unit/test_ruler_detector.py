import pytest
import numpy as np
import cv2
from pathlib import Path
import logging # For checking log messages if needed in more advanced tests
from typing import Tuple

from esnf_mat_analyzer.processing.ruler_detector import RulerDetector
from esnf_mat_analyzer.core.data_types import RulerDetectionConfig

# (default_ruler_config and ruler_detector fixtures remain the same)

@pytest.fixture
def default_ruler_config() -> RulerDetectionConfig:
    """Returns a default RulerDetectionConfig."""
    return RulerDetectionConfig(
        min_line_length=30,
        max_line_gap=5,
        hough_threshold=15, # Adjusted for potentially clearer test images
        expected_tick_distance_mm=10.0 # Default for tests, can be overridden
    )

@pytest.fixture
def ruler_detector(default_ruler_config: RulerDetectionConfig) -> RulerDetector:
    """Returns a RulerDetector instance initialized with default config."""
    return RulerDetector(config=default_ruler_config)

def create_test_ruler_image(
    width: int, height: int,
    ruler_y_top: int, ruler_thickness: int,
    tick_start_x: int, num_major_ticks: int,
    pixels_per_major_tick: int,
    tick_extension: int = 10, # How much ticks extend above/below ruler body
    tick_width: int = 1,
    ruler_color: Tuple[int,int,int] = (20,20,20),
    tick_color: Tuple[int,int,int] = (10,10,10),
    background_color: Tuple[int,int,int] = (230,230,230)
    ) -> np.ndarray:
    """
    Generates a test image with a horizontal ruler.
    Args:
        ruler_y_top: Y-coordinate of the top edge of the ruler body.
        ruler_thickness: Thickness of the ruler body in pixels.
        tick_extension: How many pixels ticks extend above top and below bottom edge.
    """
    img = np.full((height, width, 3), background_color, dtype=np.uint8)

    ruler_y_bottom = ruler_y_top + ruler_thickness

    # Ruler body
    cv2.line(img, (tick_start_x, ruler_y_top), (tick_start_x + (num_major_ticks-1)*pixels_per_major_tick, ruler_y_top), ruler_color, 2)
    cv2.line(img, (tick_start_x, ruler_y_bottom), (tick_start_x + (num_major_ticks-1)*pixels_per_major_tick, ruler_y_bottom), ruler_color, 2)

    tick_y_start = ruler_y_top - tick_extension
    tick_y_end = ruler_y_bottom + tick_extension

    for i in range(num_major_ticks):
        x = tick_start_x + i * pixels_per_major_tick
        cv2.line(img, (x, tick_y_start), (x, tick_y_end), tick_color, tick_width)
    return img


class TestRulerDetector:
    """Test suite for the RulerDetector class."""

    def test_initialization(self, ruler_detector: RulerDetector):
        assert ruler_detector is not None
        assert ruler_detector.logger is not None
        assert isinstance(ruler_detector.config, RulerDetectionConfig)

    def test_detect_scale_no_ruler_present(self, ruler_detector: RulerDetector):
        blank_image = np.full((480, 640, 3), (200,200,200), dtype=np.uint8) # Use a non-black blank
        scale = ruler_detector.detect_scale(blank_image)
        assert scale is None, "Scale should be None if no ruler is detected"

    def test_detect_scale_with_config_disabled(self, ruler_detector: RulerDetector):
        # Get the default config from the fixture and modify it
        config = ruler_detector.config # Uses the one from default_ruler_config fixture
        original_state = config.enabled
        config.enabled = False
        # No need to create a new detector, just change config state if detector uses it dynamically
        # Or, create a new one:
        # detector_disabled = RulerDetector(config=RulerDetectionConfig(enabled=False))

        dummy_image = create_test_ruler_image(400,300, 140,20, 70,5,50,15)

        # If config is checked at start of detect_scale, this should be None
        scale = ruler_detector.detect_scale(dummy_image)
        assert scale is None
        config.enabled = original_state # Reset for other tests using the same fixture instance

    def test_detect_scale_clear_horizontal_ruler(self, default_ruler_config: RulerDetectionConfig):
        img_width, img_height = 500, 300
        ruler_y = 140
        ruler_thickness = 20
        tick_start = 70
        num_ticks = 5  # Creates 4 intervals
        pixels_between_ticks = 75

        default_ruler_config.expected_tick_distance_mm = 10.0 # e.g., 1cm between major ticks
        # Override other params if needed for this specific test image
        default_ruler_config.min_line_length = 40
        default_ruler_config.hough_threshold = 20

        detector = RulerDetector(config=default_ruler_config)

        test_image = create_test_ruler_image(
            width=img_width, height=img_height,
            ruler_y_top=ruler_y, ruler_thickness=ruler_thickness,
            tick_start_x=tick_start, num_major_ticks=num_ticks,
            pixels_per_major_tick=pixels_between_ticks,
            tick_extension=15, tick_width=2
        )
        # cv2.imwrite("debug_test_clear_ruler.png", test_image) # For visual inspection

        scale = detector.detect_scale(test_image)

        assert scale is not None, "Scale should be detected for a clear ruler"
        expected_scale = pixels_between_ticks / default_ruler_config.expected_tick_distance_mm
        assert abs(scale - expected_scale) < 0.1, f"Calculated scale {scale} is not close to expected {expected_scale}"

    def test_detect_scale_too_few_ticks(self, default_ruler_config: RulerDetectionConfig):
        """Test with a ruler that has too few ticks to be reliable."""
        img_width, img_height = 400, 200
        pixels_between_ticks = 60
        default_ruler_config.expected_tick_distance_mm = 5.0

        detector = RulerDetector(config=default_ruler_config)

        # Only 2 ticks, forming 1 interval. _calculate_scale_from_ticks expects min_reliable_intervals = 2.
        test_image = create_test_ruler_image(img_width,img_height, 80,15, 50, 2, pixels_between_ticks, 10)

        scale = detector.detect_scale(test_image)
        assert scale is None, "Scale should be None if too few tick intervals are found"

    def test_detect_scale_high_tick_variability(self, default_ruler_config: RulerDetectionConfig, caplog):
        """Test with a ruler where tick spacing is too variable."""
        img_width, img_height = 600, 300
        default_ruler_config.expected_tick_distance_mm = 10.0
        detector = RulerDetector(config=default_ruler_config)

        # Create image with irregular ticks
        img = np.full((img_height, img_width, 3), (230,230,230), dtype=np.uint8)
        r_y_top, r_thick = 140, 20
        r_y_bottom = r_y_top + r_thick
        cv2.line(img, (50, r_y_top), (550, r_y_top), (20,20,20), 2)
        cv2.line(img, (50, r_y_bottom), (550, r_y_bottom), (20,20,20), 2)

        tick_y_s, tick_y_e = r_y_top - 15, r_y_bottom + 15
        # Irregular spacings: 50, 55, 45, 80, 70
        ticks_x = [100, 150, 205, 250, 330, 400]
        for x_coord in ticks_x:
            cv2.line(img, (x_coord, tick_y_s), (x_coord, tick_y_e), (10,10,10), 2)

        # The CoV check might return a scale but log a warning, or return None if strict.
        # Current _calculate_scale_from_ticks logs warning if CoV > 0.25 but still returns scale.
        # Let's ensure it returns a scale (median of [50,55,45,80,70] = 55)
        # and check for the warning.
        with caplog.at_level(logging.WARNING):
            scale = detector.detect_scale(img)

        assert scale is not None, "Scale should still be calculated based on median even with variability"
        expected_median_spacing = 55.0 # Median of [50, 55, 45, 80, 70]
        expected_scale = expected_median_spacing / default_ruler_config.expected_tick_distance_mm
        assert abs(scale - expected_scale) < 0.1, f"Scale {scale} not based on median {expected_scale}"
