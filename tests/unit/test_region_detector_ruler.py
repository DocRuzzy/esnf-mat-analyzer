"""Regression tests for FourRegionDetector ruler masking.

Historical bug: a fixed 50 px edge margin plus a global-std contrast test
marked the ENTIRE perimeter (~56% of a ~230 px image) as ruler — the mat
rim tripped the std>40 test in all four edge strips. Background coverage
in the bottom half collapsed to 1-6%, starving the polynomial background
fit (which then extrapolated to negative values under the mat).

The fix: image-scaled margin, structure-based tick detection (many
bright/dark transitions per line — p90 >= 20), and a safety valve that
drops the heuristic ruler mask if background coverage collapses below 5%.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from esnf_mat_analyzer.processing.region_detector import FourRegionDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests" / "Samples"

RULER_SAMPLE = SAMPLES / "TCD6-GPED2.0-1.png"      # ruler at bottom
NO_RULER_SAMPLE = SAMPLES / "TCD10-GPED2.0-1.png"  # no ruler in frame


def _synthetic_mat_with_ruler(h=230, w=228):
    """Small dark-background image: bright mat disc + tick ruler at bottom."""
    rng = np.random.default_rng(11)
    img = np.full((h, w), 90.0) + rng.normal(0, 5, (h, w))

    y, x = np.ogrid[:h, :w]
    mat = (x - w // 2) ** 2 + (y - h // 2 + 15) ** 2 <= 55 ** 2
    img[mat] = 225.0 + rng.normal(0, 8, int(mat.sum()))

    # Ruler: light strip with dark ticks every 4 px in the bottom 20 rows
    img[-20:, :] = 190.0
    for x0 in range(2, w, 4):
        img[-16:-4, x0:x0 + 2] = 40.0

    return np.clip(img, 0, 255).astype(np.uint8)


class TestSyntheticRuler:
    def test_ruler_limited_to_bottom_strip(self):
        img = _synthetic_mat_with_ruler()
        det = FourRegionDetector().detect(img)
        h, w = img.shape
        ruler_pct = 100.0 * det.ruler_mask.sum() / (h * w)
        assert det.ruler_detected
        assert ruler_pct < 25.0, f"ruler mask covers {ruler_pct:.0f}% (was ~56%)"
        # The mask must not touch the top half (historically the whole
        # perimeter fired).
        assert not det.ruler_mask[: h // 2, :].any()

    def test_background_coverage_not_starved(self):
        img = _synthetic_mat_with_ruler()
        det = FourRegionDetector().detect(img)
        h = img.shape[0]
        bg = det.background_mask.astype(bool)
        # Bottom half must retain usable background samples (was 1-6%).
        assert 100.0 * bg[h // 2:, :].mean() > 15.0

    def test_mat_rim_does_not_trip_side_edges(self):
        """A mat blob crossing an edge strip is ~2 transitions per line,
        far below the tick threshold."""
        img = _synthetic_mat_with_ruler()
        det = FourRegionDetector()
        h, w = img.shape
        m = max(5, int(0.08 * min(h, w)))
        assert not det._is_ruler_region(img[:, :m])       # left
        assert not det._is_ruler_region(img[:, -m:])      # right
        assert not det._is_ruler_region(img[:m, :])       # top
        assert det._is_ruler_region(img[-m:, :])          # bottom (real ruler)


@pytest.mark.skipif(not RULER_SAMPLE.exists(), reason="sample not available")
class TestRealImages:
    def test_real_ruler_detected_and_tight(self):
        gray = cv2.imread(str(RULER_SAMPLE), cv2.IMREAD_GRAYSCALE)
        det = FourRegionDetector().detect(gray)
        h, w = gray.shape
        assert det.ruler_detected
        assert 100.0 * det.ruler_mask.sum() / (h * w) < 25.0

    @pytest.mark.skipif(not NO_RULER_SAMPLE.exists(), reason="sample not available")
    def test_no_ruler_image_not_flagged(self):
        gray = cv2.imread(str(NO_RULER_SAMPLE), cv2.IMREAD_GRAYSCALE)
        det = FourRegionDetector().detect(gray)
        assert not det.ruler_detected

    def test_background_bottom_coverage_restored(self):
        gray = cv2.imread(str(RULER_SAMPLE), cv2.IMREAD_GRAYSCALE)
        det = FourRegionDetector().detect(gray)
        h = gray.shape[0]
        bg = det.background_mask.astype(bool)
        assert 100.0 * bg[h // 2:, :].mean() > 15.0, (
            "bottom-half background coverage collapsed (was 1-6% pre-fix)"
        )
