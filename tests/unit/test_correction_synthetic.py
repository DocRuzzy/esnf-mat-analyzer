"""Ground-truth validation of background correction on synthetic mats.

Uses SyntheticMatGenerator's additive illumination field (gradient +
reflection blobs) to verify that the subtraction-based correction:

1. Flattens the background back to near its noise floor
2. Removes most of the injected illumination gradient
3. Preserves the mat's relative intensity structure (thickness signal)

The thickness-fidelity check compares the corrected image against an
illumination-free twin generated with the same seed, which isolates the
correction's effect from generator noise and estimator nonlinearity.
"""
import numpy as np
import pytest

from esnf_mat_analyzer.core.data_types import BackgroundCorrectionOperator
from esnf_mat_analyzer.data.synthetic_mat_generator import (
    PatternType,
    SyntheticMatConfig,
    SyntheticMatGenerator,
)
from esnf_mat_analyzer.processing.background_correction.advanced import (
    AdvancedBackgroundProcessor,
)

SEED = 42
NOISE_STD = 5.0
GRADIENT_AMPLITUDE = 30.0


def _make_config(**overrides) -> SyntheticMatConfig:
    base = dict(
        width=400,
        height=400,
        background_level=40.0,
        background_noise_std=NOISE_STD,
        pattern_type=PatternType.RADIAL_GRADIENT,
        saturation_enabled=False,
        seed=SEED,
    )
    base.update(overrides)
    return SyntheticMatConfig(**base)


@pytest.fixture()
def lit_result():
    """Synthetic mat WITH illumination gradient + reflection blobs."""
    cfg = _make_config(
        illumination_gradient_enabled=True,
        illumination_gradient_amplitude=GRADIENT_AMPLITUDE,
        illumination_gradient_angle_deg=30.0,
        reflection_blobs=2,
    )
    return SyntheticMatGenerator(cfg).generate()


@pytest.fixture()
def gradient_only_result():
    """Synthetic mat with ONLY the smooth gradient (no blobs).

    The noise-floor flattening criterion applies to what an order-2
    polynomial can actually model; local Gaussian blobs are outside its
    model capacity and are covered by a separate improvement test.
    """
    cfg = _make_config(
        illumination_gradient_enabled=True,
        illumination_gradient_amplitude=GRADIENT_AMPLITUDE,
        illumination_gradient_angle_deg=30.0,
        reflection_blobs=0,
    )
    return SyntheticMatGenerator(cfg).generate()


@pytest.fixture()
def flat_result():
    """Illumination-free twin (same seed, no gradient/blobs)."""
    cfg = _make_config()
    return SyntheticMatGenerator(cfg).generate()


def _correct(result, operator=BackgroundCorrectionOperator.SUBTRACT):
    proc = AdvancedBackgroundProcessor()
    est = proc._step2a_polynomial_surface_fitting(
        result.image, result.background_mask, 2
    )
    corrected = proc._step3_correct_image(
        result.image, est,
        background_mask=result.background_mask,
        operator=operator,
    )
    return corrected, proc.last_correction_quality


class TestIlluminationField:
    def test_field_is_stored_and_additive(self, lit_result, flat_result):
        assert lit_result.illumination_field is not None
        assert lit_result.illumination_field.shape == lit_result.image.shape
        # The lit image should be brighter than the flat twin where the
        # field is strong (up to clipping).
        field = lit_result.illumination_field
        strong = field > 0.75 * field.max()
        diff = (lit_result.image.astype(float) - flat_result.image.astype(float))
        assert diff[strong].mean() > 0.5 * field[strong].mean()

    def test_flat_twin_has_no_field(self, flat_result):
        assert flat_result.illumination_field is None


class TestSubtractionCorrection:
    def test_background_flattened_to_noise_floor(self, gradient_only_result):
        corrected, _ = _correct(gradient_only_result)
        bg = corrected[gradient_only_result.background_mask].astype(float)
        # Injected gradient (30 DN) must be gone: background std should
        # return to <= 1.2x the pure noise level.
        assert bg.std() <= 1.2 * NOISE_STD, (
            f"background std {bg.std():.1f} vs noise floor {NOISE_STD}"
        )

    def test_blobs_improve_background(self, lit_result):
        """Local reflection blobs exceed order-2 model capacity, but the
        correction must still IMPROVE background flatness, never worsen it."""
        corrected, _ = _correct(lit_result)
        bgm = lit_result.background_mask
        before = lit_result.image[bgm].astype(float).std()
        after = corrected[bgm].astype(float).std()
        assert after < before, f"bg std worsened: {before:.1f} -> {after:.1f}"

    def test_residual_gradient_below_20pct(self, lit_result):
        corrected, _ = _correct(lit_result)
        bgm = lit_result.background_mask
        ys, xs = np.where(bgm)
        vals = corrected[bgm].astype(float)
        h, w = corrected.shape
        # Fit a plane to corrected background; its amplitude across the
        # image measures the residual (uncorrected) gradient.
        A = np.column_stack([xs / w, ys / h, np.ones_like(xs, dtype=float)])
        coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)
        residual_amplitude = abs(coeffs[0]) + abs(coeffs[1])
        assert residual_amplitude < 0.2 * GRADIENT_AMPLITUDE, (
            f"residual gradient {residual_amplitude:.1f} DN "
            f"vs injected {GRADIENT_AMPLITUDE} DN"
        )

    def test_no_new_saturation(self, lit_result):
        corrected, quality = _correct(lit_result)
        assert quality["new_saturation_pct"] <= 0.1

    def test_mat_structure_preserved_vs_flat_twin(self, lit_result, flat_result):
        """Corrected lit image ~ flat twin inside the mat (r>=0.98)."""
        corrected, _ = _correct(lit_result)
        matm = lit_result.mat_mask & flat_result.mat_mask
        a = corrected[matm].astype(float)
        b = flat_result.image[matm].astype(float)
        r = np.corrcoef(a, b)[0, 1]
        assert r >= 0.98, f"mat correlation r={r:.4f}"
        # Levels may differ by a constant offset (min-offset subtraction);
        # after removing the offset, RMSE should be small vs signal range.
        rmse = np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2))
        signal_range = np.percentile(b, 99) - np.percentile(b, 1)
        assert rmse <= 0.05 * signal_range, (
            f"offset-free RMSE {rmse:.1f} vs 5% of range {0.05 * signal_range:.1f}"
        )


class TestDivisionLegacy:
    def test_divide_operator_still_available(self, lit_result):
        corrected, quality = _correct(
            lit_result, operator=BackgroundCorrectionOperator.DIVIDE
        )
        assert quality["operator"] == "divide"
        assert corrected.dtype == np.uint8
