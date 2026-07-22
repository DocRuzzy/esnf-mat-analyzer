"""Regression: the image must be background-corrected exactly ONCE.

Historical bug: MatAnalyzer.process_image() ran ImageProcessor.preprocess()
(which applied the old leveling correction via complete_uniformity_analysis)
and THEN applied apply_background_correction_with_exclusion on top — the
image was corrected twice whenever a non-NONE method was selected.

The fix: preprocess(raw_image, apply_leveling=False) in the analyzer, since
the dedicated four-region correction always follows.
"""
from unittest.mock import patch

import numpy as np
import pytest

from esnf_mat_analyzer.core.data_types import (
    BackgroundCorrectionMethod,
    ProcessingConfig,
)
from esnf_mat_analyzer.processing.image_processor import ImageProcessor


@pytest.fixture()
def config():
    cfg = ProcessingConfig()
    cfg.background_correction_method = BackgroundCorrectionMethod.POLYNOMIAL_SURFACE
    cfg.leveling.enabled = True
    return cfg


@pytest.fixture()
def rgb_image():
    rng = np.random.default_rng(3)
    img = rng.integers(20, 60, (120, 120, 3), dtype=np.uint8)
    img[40:80, 40:80] = 210  # bright "mat" block
    return img


def test_apply_leveling_false_skips_old_correction(config, rgb_image):
    processor = ImageProcessor(config)
    with patch.object(
        processor.bg_processor, "complete_uniformity_analysis"
    ) as spy:
        processor.preprocess(rgb_image, apply_leveling=False)
        spy.assert_not_called()


def test_apply_leveling_default_still_runs_old_path(config, rgb_image):
    """Default preprocess keeps legacy behavior for standalone callers."""
    processor = ImageProcessor(config)
    with patch.object(
        processor.bg_processor, "complete_uniformity_analysis",
        return_value={"corrected_image": np.zeros((120, 120), dtype=np.uint8)},
    ) as spy:
        processor.preprocess(rgb_image)
        spy.assert_called_once()


def test_analyzer_preprocess_call_uses_apply_leveling_false():
    """The analyzer source must pass apply_leveling=False (single correction)."""
    import inspect

    from esnf_mat_analyzer.core import analyzer as analyzer_module

    source = inspect.getsource(analyzer_module)
    assert "apply_leveling=False" in source, (
        "analyzer.process_image must call preprocess(..., apply_leveling=False) "
        "to avoid double background correction"
    )
