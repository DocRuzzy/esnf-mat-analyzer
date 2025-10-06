import time
import numpy as np
from pathlib import Path

from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.config.config_manager import get_default_config
from esnf_mat_analyzer.validation.synthetic_ruler_generator import SyntheticRulerGenerator


def test_ci_pipeline_runs_quickly(tmp_path):
    """Run a minimal full-pipeline integration test on a small synthetic image.

    Purpose: ensure no import/runtime errors in the full pipeline and metrics are produced.
    This test uses a small image (256x256) and keeps runtime small.
    """
    # Generate synthetic image (small)
    gen = SyntheticRulerGenerator(width=256, height=256)
    image = gen.generate_ruler(ruler_thickness=8, rotation_angle=0)

    # Save temporary image file
    img_path = tmp_path / "synthetic_ruler.png"
    try:
        import cv2
        cv2.imwrite(str(img_path), image)
    except Exception:
        # Fallback: use PIL
        from PIL import Image
        Image.fromarray(image).save(img_path)

    config = get_default_config()
    # Reduce heavy settings for CI test
    config.processing.leveling.enabled = False
    config.ruler_detection.enabled = True

    analyzer = NanoFiberAnalyzer.from_config(config)

    start = time.time()
    result = analyzer.process_image(img_path)
    duration = time.time() - start

    # Basic assertions
    assert hasattr(result, 'metrics')
    assert isinstance(result.metrics, dict)
    # Ensure it ran quickly on CI (< 30s is reasonable for small image)
    assert duration < 30.0
    # At least one metric should be present
    assert len(result.metrics) > 0
