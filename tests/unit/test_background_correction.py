"""Pytest coverage for the background-correction Step 3 operator.

Historical note (why subtraction is back)
-----------------------------------------
An early "subtraction" method was labeled broken and replaced by division.
That method was `cv2.subtract(image, median_blur_background)` followed by
`cv2.bitwise_not` — it clipped the difference at 0 AND inverted polarity
(white mat became dark), which is why it was abandoned.

The current SUBTRACT operator is a different formulation:

    corrected = image - (illumination_model - illumination_model.min())

It subtracts only the spatial VARIATION of the modeled illumination.
Polarity is preserved (mat stays bright), the subtracted amount is always
>= 0 so no pixel can brighten (new saturation is impossible), and there is
no clipping mass at 0 for realistic models. Division, by contrast, explodes
where the model approaches zero — on the black-background mat photos it
blew out 23/30 sample images (up to 98% of mat pixels clipped to white).

Regression anchors below use tests/Samples/TCD6-GPED2.0-1.png, the worst
historical division blow-out (96.5% new mat saturation).
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from esnf_mat_analyzer.core.data_types import BackgroundCorrectionOperator
from esnf_mat_analyzer.processing.background_correction.advanced import (
    AdvancedBackgroundProcessor,
)
from esnf_mat_analyzer.processing.region_detector import FourRegionDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
WORST_SAMPLE = REPO_ROOT / "tests" / "Samples" / "TCD6-GPED2.0-1.png"


def _synthetic_gradient_scene(seed=7):
    """Dark background + bright mat disc + linear illumination gradient."""
    rng = np.random.default_rng(seed)
    h, w = 300, 300
    image = np.full((h, w), 40.0)
    image += rng.normal(0, 4.0, (h, w))

    y, x = np.ogrid[:h, :w]
    mat_mask = (x - w // 2) ** 2 + (y - h // 2) ** 2 <= 60 ** 2
    image[mat_mask] = 200.0 + rng.normal(0, 8.0, int(mat_mask.sum()))

    gradient = (np.arange(w, dtype=float) / (w - 1)) * 30.0  # 30 DN left->right
    image += gradient[None, :]

    background_mask = ~mat_mask
    return np.clip(image, 0, 255).astype(np.uint8), mat_mask, background_mask


@pytest.fixture()
def gradient_scene():
    return _synthetic_gradient_scene()


class TestSubtractOperator:
    def test_flattens_gradient_preserves_mat(self, gradient_scene):
        image, mat_mask, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        corrected = proc._step3_correct_image(
            image, est, background_mask=bg_mask,
            operator=BackgroundCorrectionOperator.SUBTRACT,
        )
        # Gradient removed: background std back near noise level
        assert corrected[bg_mask].std() < 0.6 * image[bg_mask].std()
        # Polarity preserved: mat stays bright relative to background
        assert corrected[mat_mask].mean() > corrected[bg_mask].mean() + 50

    def test_no_clipping_mass_at_zero(self, gradient_scene):
        image, _, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        corrected = proc._step3_correct_image(image, est, background_mask=bg_mask)
        assert np.mean(corrected == 0) < 0.001  # <0.1% of pixels at 0

    def test_no_new_saturation_possible(self, gradient_scene):
        """Min-offset subtraction can never brighten a pixel."""
        image, _, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        corrected = proc._step3_correct_image(image, est, background_mask=bg_mask)
        assert np.all(corrected.astype(int) <= image.astype(int) + 1)  # +1 rounding

    def test_negative_model_is_clipped_not_fatal(self, gradient_scene):
        """A model dipping below zero (starved fit) must be handled safely."""
        image, mat_mask, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        est_negative = est - (est.min() + 25.0)  # force min to -25
        corrected = proc._step3_correct_image(
            image, est_negative, background_mask=bg_mask
        )
        assert corrected.dtype == np.uint8
        # Mat must NOT be blown out (this is where division exploded)
        assert np.mean(corrected[mat_mask] >= 254) < 0.05


class TestDivideOperatorLegacy:
    def test_divide_reproduces_legacy_behavior(self, gradient_scene):
        """DIVIDE must remain available and match the legacy formula."""
        image, _, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        corrected = proc._step3_correct_image(
            image, est, background_mask=bg_mask,
            operator=BackgroundCorrectionOperator.DIVIDE,
        )
        expected = np.clip(
            image.astype(np.float32) / np.maximum(est.astype(np.float32), 1.0)
            * float(np.mean(est)),
            0, 255,
        ).astype(np.uint8)
        assert np.array_equal(corrected, expected)

    def test_quality_metrics_reported(self, gradient_scene):
        image, _, bg_mask = gradient_scene
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(image, bg_mask, 2)
        proc._step3_correct_image(image, est, background_mask=bg_mask,
                                  operator=BackgroundCorrectionOperator.DIVIDE)
        q = proc.last_correction_quality
        assert q["operator"] == "divide"
        assert {"new_saturation_pct", "detail_loss_pct", "variance_ratio"} <= set(q)


@pytest.mark.skipif(not WORST_SAMPLE.exists(), reason="sample image not available")
class TestRealImageRegression:
    """TCD6-GPED2.0-1.png: the worst historical division blow-out."""

    def _run(self, operator):
        gray = cv2.imread(str(WORST_SAMPLE), cv2.IMREAD_GRAYSCALE)
        det = FourRegionDetector().detect(gray)
        bgm = det.background_mask.astype(bool)
        matm = det.mat_mask.astype(bool)
        proc = AdvancedBackgroundProcessor()
        est = proc._step2a_polynomial_surface_fitting(gray, bgm, 2)
        corrected = proc._step3_correct_image(
            gray, est, background_mask=bgm, operator=operator
        )
        return gray, corrected, matm, bgm

    def test_subtract_does_not_blow_out_mat(self):
        gray, corrected, matm, bgm = self._run(BackgroundCorrectionOperator.SUBTRACT)
        sat_before = 100.0 * np.sum(gray[matm] >= 254) / matm.sum()
        sat_after = 100.0 * np.sum(corrected[matm] >= 254) / matm.sum()
        assert sat_after - sat_before < 1.0, (
            f"new mat saturation {sat_after - sat_before:.1f}pp "
            f"(division historically produced +96.5pp here)"
        )

    def test_subtract_preserves_mat_detail(self):
        gray, corrected, matm, _ = self._run(BackgroundCorrectionOperator.SUBTRACT)
        assert corrected[matm].std() >= 0.9 * gray[matm].std()

    def test_subtract_flattens_background(self):
        gray, corrected, _, bgm = self._run(BackgroundCorrectionOperator.SUBTRACT)
        assert corrected[bgm].std() <= gray[bgm].std()
