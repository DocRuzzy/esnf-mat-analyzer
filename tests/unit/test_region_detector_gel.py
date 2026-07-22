"""Regression tests for gel-region and mat-candidate detection.

Two historical failure modes on real samples (2026-07-22):

1. Gel detection used a pure darkness threshold inside a dilated ring —
   it swallowed the diffuse shadow below the mat (darker than background
   but not gel) and missed thin gel arcs that were not dark enough.
   Fixed by radial ray-casting with per-ray local background
   (_detect_gel_region_radial): the gel is the annulus between the mat
   edge and the last deep-dark sample along each ray.

2. Mat selection took the LARGEST bright component — on TCD10-GPED3.0-1
   a bright glare band across the top of the background (9134 px)
   outgrew the mat disc (8677 px) and was selected as "mat".
   Fixed by scoring candidates with area x circularity x centrality.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from esnf_mat_analyzer.processing.region_detector import FourRegionDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests" / "Samples"

GEL_SAMPLE = SAMPLES / "TCD4-GPED3.0-1.png"      # clear gel ring + shadow below
GLARE_SAMPLE = SAMPLES / "TCD10-GPED3.0-1.png"   # glare band larger than mat


@pytest.mark.skipif(not GEL_SAMPLE.exists(), reason="sample not available")
class TestGelRing:
    @pytest.fixture()
    def result(self):
        gray = cv2.imread(str(GEL_SAMPLE), cv2.IMREAD_GRAYSCALE)
        return gray, FourRegionDetector().detect(gray)

    def test_gel_detected_and_darker_than_background(self, result):
        gray, res = result
        assert res.gel_detected
        assert res.gel_intensity < res.background_intensity

    def test_gel_is_annulus_hugging_mat(self, result):
        """Every gel pixel must be close to the mat (it is a contact ring),
        which excludes the detached shadow blob the old method included."""
        gray, res = result
        mat = res.mat_mask.astype(np.uint8)
        r_typ = np.sqrt(mat.sum() / np.pi)
        reach = int(0.5 * r_typ)
        near_mat = cv2.dilate(
            mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
        ).astype(bool)
        gel = res.gel_mask.astype(bool)
        assert gel.sum() > 0
        outside = gel & ~near_mat
        assert outside.sum() < 0.02 * gel.sum(), (
            f"{outside.sum()} gel px far from mat (shadow leakage)"
        )

    def test_gel_not_shadow_blob(self, result):
        """The old threshold method returned ~6800 px (ring + detached
        shadow blob). The ring including its faint outer rim is ~5600 px;
        anything approaching 6800 means the shadow is back."""
        gray, res = result
        assert res.gel_mask.sum() < 6200

    def test_gel_surrounds_mat(self, result):
        """Gel must be present around most of the mat (>60% of directions),
        not just the dark bottom arc."""
        gray, res = result
        mat = res.mat_mask.astype(bool)
        gel = res.gel_mask.astype(bool)
        ys, xs = np.nonzero(mat)
        cy, cx = ys.mean(), xs.mean()
        gys, gxs = np.nonzero(gel)
        angles = np.arctan2(gys - cy, gxs - cx)
        bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))[0]
        coverage = np.mean(bins > 0)
        assert coverage > 0.6, f"gel covers only {coverage:.0%} of directions"


RING_SAMPLE = SAMPLES / "TCD6-GPED2.0-1.png"  # complete ring + mat spill outside gel


@pytest.mark.skipif(not RING_SAMPLE.exists(), reason="sample not available")
class TestCompleteGelRing:
    """User-reported (2026-07-22): on TCD6-GPED2.0-1 the gel forms a
    complete ring around the mat, with spilled mat OUTSIDE the gel at
    4:30-6:00 clock direction. Early radial detection missed the outer
    gel band (only 10-20 DN below local background), the gel behind the
    spill (rays started at the outermost mat pixel), and the bottom arc
    (gel indistinguishable from deep shadow -> now interpolated from
    angular neighbours)."""

    @pytest.fixture()
    def result(self):
        gray = cv2.imread(str(RING_SAMPLE), cv2.IMREAD_GRAYSCALE)
        return gray, FourRegionDetector().detect(gray)

    def test_ring_is_complete(self, result):
        gray, res = result
        mat = res.mat_mask.astype(bool)
        gel = res.gel_mask.astype(bool)
        ys, xs = np.nonzero(mat)
        cy, cx = ys.mean(), xs.mean()
        gys, gxs = np.nonzero(gel)
        angles = np.arctan2(gys - cy, gxs - cx)
        bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))[0]
        coverage = np.mean(bins > 0)
        assert coverage >= 0.9, f"gel ring covers only {coverage:.0%} of directions"

    def test_gel_not_patchy(self, result):
        """Patchy detection found only ~2700 px; the complete ring is far larger."""
        gray, res = result
        assert res.gel_mask.sum() > 4000


@pytest.mark.skipif(not GLARE_SAMPLE.exists(), reason="sample not available")
class TestMatSelection:
    def test_mat_is_central_disc_not_glare_band(self):
        gray = cv2.imread(str(GLARE_SAMPLE), cv2.IMREAD_GRAYSCALE)
        res = FourRegionDetector().detect(gray)
        h, w = gray.shape
        mat = res.mat_mask.astype(bool)
        ys, xs = np.nonzero(mat)
        cy, cx = ys.mean(), xs.mean()
        # Centroid near the image center (the glare band centroid was at y~20)
        assert abs(cy - h / 2) < 0.15 * h, f"mat centroid y={cy:.0f} (h={h})"
        assert abs(cx - w / 2) < 0.15 * w
        # The glare band spanned the full top edge; the disc must not
        assert not mat[0, :].any(), "mat touches the top image border"

    def test_synthetic_band_vs_disc(self):
        """Synthetic: full-width bright band LARGER than the disc must lose."""
        rng = np.random.default_rng(5)
        h, w = 220, 220
        img = np.full((h, w), 70.0) + rng.normal(0, 4, (h, w))
        img[:45, :] = 150.0  # glare band: 9900 px
        y, x = np.ogrid[:h, :w]
        disc = (x - 110) ** 2 + (y - 130) ** 2 <= 50 ** 2  # disc: ~7850 px
        img[disc] = 200.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        res = FourRegionDetector(ruler_detection_enabled=False).detect(img)
        mat = res.mat_mask.astype(bool)
        overlap = (mat & disc).sum() / disc.sum()
        assert overlap > 0.8, f"mat overlaps disc only {overlap:.0%}"
        assert not mat[:30, :].any(), "mat grabbed the glare band"
