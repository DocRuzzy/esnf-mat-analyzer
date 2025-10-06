"""Helpers to score background correction results and recommend a method.

The composite metric balances: lower saturation rate, preserved dynamic range,
and preservation of signal within a provided ROI (if any).
"""
from typing import Optional, Callable, Dict, Tuple
import numpy as np


def _saturation_rate(image: np.ndarray, sat_thresh: int = 240) -> float:
    if image is None or image.size == 0:
        return 1.0
    return float(np.sum(image >= sat_thresh)) / image.size


def _dynamic_range_preservation(orig: np.ndarray, corr: np.ndarray) -> float:
    # Dynamic range as 98th-2nd percentiles
    try:
        o_range = np.percentile(orig, 98) - np.percentile(orig, 2)
        c_range = np.percentile(corr, 98) - np.percentile(corr, 2)
        if o_range <= 0:
            return 0.0
        return float(max(0.0, min(1.0, c_range / o_range)))
    except Exception:
        return 0.0


def _signal_preservation(orig: np.ndarray, corr: np.ndarray, mask: Optional[np.ndarray]) -> float:
    try:
        if mask is None:
            # Use center region heuristic
            h, w = orig.shape[:2]
            y0, x0 = h // 4, w // 4
            y1, x1 = 3 * h // 4, 3 * w // 4
            o = orig[y0:y1, x0:x1]
            c = corr[y0:y1, x0:x1]
        else:
            o = orig[mask]
            c = corr[mask]

        if o.size == 0:
            return 0.0
        # Compare mean intensities (preserve relative signal)
        o_mean = float(np.mean(o))
        c_mean = float(np.mean(c))
        if o_mean == 0:
            return 0.0
        return float(max(0.0, min(1.0, c_mean / o_mean)))
    except Exception:
        return 0.0


def score_correction(orig: np.ndarray, corr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return a composite score in [0,1] where higher is better.

    Components:
    - saturation reduction: prefer lower saturation in corrected image
    - dynamic range preservation: ratio of dynamic ranges (0..1)
    - signal preservation: mean intensity preservation in ROI (0..1)
    Weights are chosen conservatively.
    """
    # Ensure grayscale arrays
    if orig is None or corr is None:
        return 0.0
    try:
        orig_arr = np.asarray(orig).astype(np.float32)
        corr_arr = np.asarray(corr).astype(np.float32)
    except Exception:
        return 0.0

    sat_score = 1.0 - _saturation_rate(corr_arr)
    dyn_score = _dynamic_range_preservation(orig_arr, corr_arr)
    sig_score = _signal_preservation(orig_arr, corr_arr, mask)

    # Weighted sum
    score = 0.4 * dyn_score + 0.4 * sig_score + 0.2 * sat_score
    # Clamp
    return float(max(0.0, min(1.0, score)))


def recommend_method(orig: np.ndarray, methods: Dict[str, Callable[[np.ndarray], np.ndarray]], mask: Optional[np.ndarray] = None, downsample: Optional[int] = 1) -> Tuple[str, Dict[str, float]]:
    """Apply each method to orig and score; return best method id and score map.

    Args:
        orig: input grayscale image (numpy ndarray)
        methods: mapping id->callable(image)->corrected_image
        mask: optional boolean mask selecting ROI
        downsample: integer factor to downsample (>=1)

    Returns:
        (best_method_id, scores)
    """
    if orig is None or orig.size == 0:
        return "", {}

    # Convert to ndarray
    orig_arr = np.asarray(orig)
    # Downsample for speed
    if downsample and downsample > 1:
        orig_small = orig_arr[::downsample, ::downsample]
        mask_small = mask[::downsample, ::downsample] if mask is not None else None
    else:
        orig_small = orig_arr
        mask_small = mask

    scores = {}
    for mid, func in methods.items():
        try:
            corr = func(orig_small)
            sc = score_correction(orig_small, corr, mask_small)
            scores[mid] = sc
        except Exception:
            scores[mid] = 0.0

    if not scores:
        return "", {}
    # Choose highest score
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores


def get_standard_methods() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Return a mapping of standard background correction method ids to callables.

    The returned callables accept a single grayscale numpy array and return a corrected
    grayscale numpy array. This convenience function avoids duplicating method wiring
    between GUI and CLI.
    """
    try:
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        proc = AdvancedBackgroundProcessor()

        return {
            'none': lambda img: img,
            'basic': lambda img: proc.basic_correction(img, 1),
            'rolling_ball': lambda img: proc.rolling_ball_3d(img, 50),
            'restore': lambda img: proc.restore_method(img, 5.0),
            'homomorphic': lambda img: proc.homomorphic_filter(img, 30, 0.5, 2.0)
        }
    except Exception:
        # If advanced processor is unavailable, provide simple fallbacks
        return {
            'none': lambda img: img,
            'basic': lambda img: img,
            'rolling_ball': lambda img: img,
            'restore': lambda img: img,
            'homomorphic': lambda img: img,
        }
