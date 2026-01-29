"""
Analysis modules for ESNF Mat Analyzer.

This module provides uniformity metrics, frequency diagnostics,
and gel boundary shape analysis capabilities.
"""

from .uniformity_metrics import (
    RadialUniformityIndex,
    GiniCoefficient,
    ThicknessRangeRatio,
)
from .frequency_diagnostics import (
    FrequencyDiagnostics,
    FrequencyAnalysisResult,
    FilteringImpactResult,
)
from .gel_boundary_analysis import (
    GelBoundaryAnalyzer,
    GelBoundaryMetrics,
    GelUniformityCorrelation,
)

__all__ = [
    "RadialUniformityIndex",
    "GiniCoefficient",
    "ThicknessRangeRatio",
    "FrequencyDiagnostics",
    "FrequencyAnalysisResult",
    "FilteringImpactResult",
    "GelBoundaryAnalyzer",
    "GelBoundaryMetrics",
    "GelUniformityCorrelation",
]
