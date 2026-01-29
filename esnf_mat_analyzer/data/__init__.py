"""
Data generation utilities for ESNF Mat Analyzer.

This module provides synthetic data generation capabilities for
validating the analysis pipeline and testing FFT filtering effects.
"""

from .synthetic_mat_generator import (
    SyntheticMatGenerator,
    SyntheticMatConfig,
    SyntheticMatResult,
    PatternType,
    generate_fft_validation_set,
)

__all__ = [
    "SyntheticMatGenerator",
    "SyntheticMatConfig", 
    "SyntheticMatResult",
    "PatternType",
    "generate_fft_validation_set",
]
