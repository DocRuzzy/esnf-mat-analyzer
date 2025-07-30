from typing import Dict, Tuple
import numpy as np

class GiniCoefficientAnalyzer:
    """Enhanced Gini coefficient analysis for thickness distribution equality.

    Implements multiple Gini variants:
    - Standard Gini: Overall thickness inequality
    - Spatial Gini: Spatial autocorrelation-weighted inequality
    - Multi-scale Gini: Scale-dependent inequality analysis
    """

    def calculate_gini_coefficient(self, thickness_map: np.ndarray,
                                 roi_mask: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive Gini coefficient metrics.

        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region

        Returns:
            Dictionary containing multiple Gini variants
        """
        # Extract thickness values within ROI
        thickness_values = thickness_map[roi_mask]
        thickness_values = thickness_values[~np.isnan(thickness_values)]

        if len(thickness_values) < 2:
            return {'standard_gini': 0.0, 'spatial_gini': 0.0, 'normalized_gini': 0.0}

        # Standard Gini coefficient
        standard_gini = self._calculate_standard_gini(thickness_values)

        # Spatial Gini with location weighting
        spatial_gini = self._calculate_spatial_gini(thickness_map, roi_mask)

        # Normalized Gini (0-1 scale)
        normalized_gini = self._normalize_gini(standard_gini, len(thickness_values))

        return {
            'standard_gini': standard_gini,
            'spatial_gini': spatial_gini,
            'normalized_gini': normalized_gini,
            'gini_uniformity_index': 1.0 - normalized_gini  # Higher = more uniform
        }

    def _calculate_standard_gini(self, values: np.ndarray) -> float:
        """Calculate standard Gini coefficient with numerical stability."""
        sorted_values = np.sort(values)
        n = len(sorted_values)

        if n == 0 or np.sum(sorted_values) == 0:
            return 0.0

        # Gini coefficient formula with numerical stability
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

    def _calculate_spatial_gini(self, thickness_map: np.ndarray,
                              roi_mask: np.ndarray) -> float:
        """Calculate spatial Gini coefficient considering local neighborhoods."""
        from scipy.ndimage import generic_filter

        # Calculate local mean in 3x3 neighborhoods
        local_means = generic_filter(thickness_map, np.nanmean, size=3)

        # Extract values and local context
        y_coords, x_coords = np.where(roi_mask)
        thickness_values = thickness_map[roi_mask]
        local_context = local_means[roi_mask]

        # Weight by spatial coherence
        spatial_weights = 1.0 / (1.0 + np.abs(thickness_values - local_context))
        weighted_values = thickness_values * spatial_weights

        return self._calculate_standard_gini(weighted_values)

    def _normalize_gini(self, gini: float, n: int) -> float:
        """Normalize the Gini coefficient to a 0-1 scale."""
        if n <= 1:
            return 0.0
        return gini * (n / (n - 1))

class ThicknessRangeAnalyzer:
    """Advanced thickness range analysis with statistical robustness."""

    def calculate_thickness_range_ratio(self, thickness_map: np.ndarray,
                                      roi_mask: np.ndarray,
                                      percentile_range: Tuple[float, float] = (5, 95)
                                     ) -> Dict[str, float]:
        """Calculate robust thickness range metrics.

        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            percentile_range: Percentile range for robust statistics

        Returns:
            Dictionary containing multiple TRR variants
        """
        thickness_values = thickness_map[roi_mask]
        thickness_values = thickness_values[~np.isnan(thickness_values)]

        if len(thickness_values) < 2:
            return {'standard_trr': 1.0, 'robust_trr': 1.0, 'iqr_trr': 1.0}

        # Standard TRR (min/max ratio)
        t_min, t_max = np.min(thickness_values), np.max(thickness_values)
        standard_trr = t_min / t_max if t_max > 0 else 1.0

        # Robust TRR using percentiles
        p_low, p_high = np.percentile(thickness_values, percentile_range)
        robust_trr = p_low / p_high if p_high > 0 else 1.0

        # Interquartile range TRR
        q1, q3 = np.percentile(thickness_values, [25, 75])
        iqr_trr = q1 / q3 if q3 > 0 else 1.0

        return {
            'standard_trr': standard_trr,
            'robust_trr': robust_trr,
            'iqr_trr': iqr_trr,
            'thickness_span': t_max - t_min,
            'robust_thickness_span': p_high - p_low
        }
