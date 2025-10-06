"""Metric validation utilities for ESNF Mat Analyzer.

Provides lightweight validation for uniformity and related metric values so the GUI
can flag out-of-range, NaN or infinite values before display or downstream use.

This module is intentionally heuristic and conservative — it does not attempt to
replace domain-specific validation but offers sensible defaults used by the GUI.
"""
from typing import Dict, Tuple, Any
import math


def _is_finite_number(v: Any) -> bool:
    try:
        return isinstance(v, (int, float)) and not math.isnan(v) and math.isfinite(v)
    except Exception:
        return False


def validate_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Validate a dictionary of metric name -> value.

    Returns a tuple (sanitized_metrics, issues) where `sanitized_metrics` contains
    the original values (unchanged) and `issues` maps metric names to a short
    human-readable problem string for any invalid entries.

    Heuristic rules applied:
    - Non-finite numbers (NaN, Inf) are marked invalid.
    - For many named metrics we expect values in [0,1]. Texture correlation is
      allowed in [-1,1]. Contrast-like metrics are allowed >= 0.
    - Unknown metric names are accepted if numeric and finite.
    """
    issues: Dict[str, str] = {}

    for name, value in list(metrics.items()):
        if not _is_finite_number(value):
            issues[name] = "non-finite or non-numeric"
            continue

        # heuristics based on metric name
        lname = name.lower()

        # texture correlation: -1..1
        if 'correlation' in lname and 'texture' in lname:
            if not (-1.0 <= float(value) <= 1.0):
                issues[name] = "expected in [-1, 1]"
            continue

        # contrast-like metrics: >= 0
        if 'contrast' in lname or 'range' in lname or 'width' in lname:
            if float(value) < 0:
                issues[name] = "expected >= 0"
            continue

        # anisotropy, uniformity, gini, scale_* and primary indices typically 0..1
        if any(token in lname for token in ('uniformity', 'anisotropy', 'gini', 'psd_', 'scale_', 'overall', 'index', 'energy', 'homogeneity')):
            if not (0.0 <= float(value) <= 1.0):
                issues[name] = "expected in [0, 1]"
            continue

        # default: accept any finite numeric value

    return metrics, issues
