import pytest
import numpy as np
import cv2
from pathlib import Path
import logging # For checking log messages if needed in more advanced tests
from typing import Tuple, Optional

try:  # pragma: no cover - optional heavy dependency
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # Catch binary / import errors
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
from unittest.mock import MagicMock

from esnf_mat_analyzer.processing.ruler_detector import RulerDetector
from esnf_mat_analyzer.core.data_types import RulerDetectionConfig
# DeepGPModule only needed when torch is available; avoid unconditional import
if _TORCH_AVAILABLE:  # pragma: no branch
    try:
        from esnf_mat_analyzer.processing.deep_gp_module import DeepGPModule  # type: ignore
    except Exception:  # pragma: no cover
        DeepGPModule = None  # type: ignore
else:
    DeepGPModule = None  # type: ignore


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
    background_color: Tuple[int,int,int] = (230,230,230),
    img: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Generates or draws on a test image with a horizontal ruler.
    Args:
        ruler_y_top: Y-coordinate of the top edge of the ruler body.
        ruler_thickness: Thickness of the ruler body in pixels.
        tick_extension: How many pixels ticks extend above top and below bottom edge.
        img: Optional image to draw on. If None, a new image is created.
    """
    if img is None:
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
        """Test that the RulerDetector initializes correctly."""
        assert ruler_detector is not None
        assert ruler_detector.logger is not None
        assert isinstance(ruler_detector.config, RulerDetectionConfig)

    def test_detect_no_ruler_present(self, ruler_detector: RulerDetector):
        """Test that detect() returns None for an image without a ruler."""
        blank_image = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
        ruler = ruler_detector.detect(blank_image)
        assert ruler is None, "Ruler object should be None if no ruler is detected"

    def test_detect_with_config_disabled(self, ruler_detector: RulerDetector):
        """Test that detect() returns None when the feature is disabled."""
        config = ruler_detector.config
        original_state = config.enabled
        config.enabled = False

        dummy_image = create_test_ruler_image(400, 300, 140, 20, 70, 5, 50, 15)
        
        ruler = ruler_detector.detect(dummy_image)
        assert ruler is None, "Ruler object should be None when detection is disabled"

        config.enabled = original_state  # Reset for other tests

    def test_detect_clear_horizontal_ruler(self, default_ruler_config: RulerDetectionConfig):
        """Test a successful detection on a clear, ideal ruler image."""
        img_width, img_height = 500, 300
        ruler_y = 140
        num_ticks = 5
        pixels_between_ticks = 75
        
        default_ruler_config.expected_tick_distance_mm = 10.0
        detector = RulerDetector(config=default_ruler_config)

        test_image = create_test_ruler_image(
            width=img_width, height=img_height, ruler_y_top=ruler_y,
            ruler_thickness=20, tick_start_x=70, num_major_ticks=num_ticks,
            pixels_per_major_tick=pixels_between_ticks, tick_extension=15
        )

        ruler = detector.detect(test_image)

        assert ruler is not None, "A Ruler object should be returned for a clear image"
        assert ruler.scale_px_per_mm is not None, "Scale should be detected"
        expected_scale = pixels_between_ticks / default_ruler_config.expected_tick_distance_mm
        assert abs(ruler.scale_px_per_mm - expected_scale) < 0.1

        assert len(ruler.ticks) == num_ticks
        assert ruler.mask is not None
        assert ruler.mask.shape == (img_height, img_width)
        assert np.sum(ruler.mask) > 0

    def test_detect_too_few_ticks(self, default_ruler_config: RulerDetectionConfig):
        """Test that detection fails if a ruler has too few ticks to be reliable."""
        default_ruler_config.expected_tick_distance_mm = 5.0
        detector = RulerDetector(config=default_ruler_config)
        
        # Creates only 1 interval, but default is min_reliable_intervals = 2
        test_image = create_test_ruler_image(400, 200, 80, 15, 50, 2, 60, 10)

        ruler = detector.detect(test_image)
        assert ruler is None, "Detection should fail if too few tick intervals are found"

    def test_detect_high_tick_variability(self, default_ruler_config: RulerDetectionConfig, caplog):
        """Test detection on a ruler with highly variable tick spacing."""
        detector = RulerDetector(config=default_ruler_config)
        img = np.full((300, 600, 3), (230, 230, 230), dtype=np.uint8)

        # Draw ruler body
        cv2.line(img, (50, 140), (550, 140), (20, 20, 20), 2)
        cv2.line(img, (50, 160), (550, 160), (20, 20, 20), 2)
        
        # Draw irregular ticks
        ticks_x = [100, 150, 205, 250, 330, 400]  # Spacings: 50, 55, 45, 80, 70
        for x in ticks_x:
            cv2.line(img, (x, 125), (x, 175), (10, 10, 10), 2)

        with caplog.at_level(logging.WARNING):
            ruler = detector.detect(img)

        assert any("High variability in tick spacing" in record.message for record in caplog.records)
        assert ruler is not None, "A ruler should still be detected based on median"
        assert ruler.scale_px_per_mm is not None
        
        expected_median_spacing = 55.0  # Median of [50, 55, 45, 80, 70]
        expected_scale = expected_median_spacing / default_ruler_config.expected_tick_distance_mm
        assert abs(ruler.scale_px_per_mm - expected_scale) < 0.1

    def test_scoring_model_selects_correct_ruler(self, default_ruler_config: RulerDetectionConfig):
        """Test that the scoring model picks the better of two ruler candidates."""
        img_width, img_height = 600, 400
        detector = RulerDetector(config=default_ruler_config)
        img = np.full((img_height, img_width, 3), (230, 230, 230), dtype=np.uint8)

        # --- Draw the BETTER ruler (longer, more ticks) ---
        good_ruler_y = 100
        create_test_ruler_image(width=img_width, height=img_height,
            ruler_y_top=good_ruler_y, ruler_thickness=15,
            tick_start_x=50, num_major_ticks=10, pixels_per_major_tick=50,
            tick_extension=10, img=img)

        # --- Draw a DECOY ruler (shorter, fewer ticks) ---
        decoy_ruler_y = 300
        create_test_ruler_image(width=img_width, height=img_height,
            ruler_y_top=decoy_ruler_y, ruler_thickness=10,
            tick_start_x=100, num_major_ticks=4, pixels_per_major_tick=40,
            tick_extension=8, img=img)

        ruler = detector.detect(img)

        assert ruler is not None, "A ruler should be detected"
        # Check that the detected ruler's y-position is close to the BETTER ruler
        detected_ruler_y = (ruler.body_lines[0][0][1] + ruler.body_lines[1][0][1]) / 2
        assert abs(detected_ruler_y - (good_ruler_y + 15/2)) < 10, \
            "The scoring model did not select the correct ruler candidate"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available for DeepGP test")
    def test_detect_with_deep_gp_mocked(self, default_ruler_config: RulerDetectionConfig, monkeypatch, caplog):
        """Test the DeepGP integration with a mocked model."""
        # 1. Mock the DeepGPModule class
        mock_model_instance = MagicMock()
        mock_model_instance.predict_gp_params.return_value = torch.tensor([[1.0, 2.0, 75.0]])

        mock_deep_gp_module = MagicMock(return_value=mock_model_instance)
        monkeypatch.setattr('esnf_mat_analyzer.processing.ruler_detector.DeepGPModule', mock_deep_gp_module)
        monkeypatch.setattr(torch, 'load', lambda path: {})

        # 2. Configure the detector to use the (mocked) DeepGP model
        config = default_ruler_config
        config.use_deep_gp = True
        config.deep_gp_model_path = "dummy/path/to/model.pth" # Path is needed to trigger loading

        detector = RulerDetector(config=config)

        # 3. Create a test image and run detection
        img_width, img_height = 500, 300
        pixels_between_ticks = 75
        test_image = create_test_ruler_image(
            width=img_width, height=img_height, ruler_y_top=140,
            ruler_thickness=20, tick_start_x=70, num_major_ticks=5,
            pixels_per_major_tick=pixels_between_ticks, tick_extension=15
        )

        with caplog.at_level(logging.INFO):
            ruler = detector.detect(test_image)

        # 4. Assertions
        assert ruler is not None, "A ruler should be detected"

        # Check that the DeepGP prediction method was called
        mock_model_instance.predict_gp_params.assert_called_once()

        # Check that the log message indicates DeepGP was used
        assert "DeepGP predicted params" in caplog.text
        assert "DeepGP integration is a placeholder" in caplog.text

        # Since the implementation falls back to the statistical method,
        # the final scale should be the same as the standard test.
        assert ruler.scale_px_per_mm is not None
        expected_scale = pixels_between_ticks / config.expected_tick_distance_mm
        assert abs(ruler.scale_px_per_mm - expected_scale) < 0.1
