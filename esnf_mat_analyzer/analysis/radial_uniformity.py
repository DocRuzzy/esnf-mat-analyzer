from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

@dataclass
class RadialAnalysisConfig:
    """Configuration for radial uniformity analysis."""
    num_radial_bins: int = 50
    angle_resolution_degrees: float = 2.0
    smoothing_window: int = 5
    outlier_threshold_std: float = 2.0
    interpolation_method: str = 'cubic'

class RadialUniformityAnalyzer:
    """Enhanced radial uniformity analysis with statistical rigor.

    Implements multiple RUI variants for comprehensive characterization:
    - Classical RUI: Coefficient of variation along radial profiles
    - Angular RUI: Variation in circumferential direction
    - Gradient RUI: Spatial derivative-based measure
    """

    def __init__(self, config: RadialAnalysisConfig):
        self.config = config

    def calculate_radial_uniformity_index(self, thickness_map: np.ndarray,
                                        center: Tuple[int, int],
                                        roi_radius: float) -> Dict[str, float]:
        """Calculate comprehensive radial uniformity metrics.

        Args:
            thickness_map: 2D thickness distribution
            center: (y, x) coordinates of analysis center
            roi_radius: Maximum radius for analysis

        Returns:
            Dictionary containing multiple RUI variants and statistics
        """
        cy, cx = center
        y_coords, x_coords = np.ogrid[:thickness_map.shape[0], :thickness_map.shape[1]]
        radius_map = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)

        # Create radial bins
        max_radius = min(roi_radius, np.max(radius_map))
        radial_bins = np.linspace(0, max_radius, self.config.num_radial_bins)

        # Calculate radial profile with statistical measures
        radial_profile = self._calculate_radial_profile(
            thickness_map, radius_map, radial_bins
        )

        # Calculate multiple RUI variants
        classical_rui = self._calculate_classical_rui(radial_profile)
        angular_rui = self._calculate_angular_rui(thickness_map, center, radial_bins)
        gradient_rui = self._calculate_gradient_rui(radial_profile, radial_bins)

        # Statistical analysis
        profile_stats = self._calculate_profile_statistics(radial_profile)

        return {
            'classical_rui': classical_rui,
            'angular_rui': angular_rui,
            'gradient_rui': gradient_rui,
            'radial_monotonicity': self._calculate_monotonicity(radial_profile),
            'edge_uniformity': self._calculate_edge_uniformity(radial_profile),
            **profile_stats
        }

    def _calculate_radial_profile(self, thickness_map: np.ndarray,
                                radius_map: np.ndarray,
                                radial_bins: np.ndarray) -> np.ndarray:
        """Calculate mean thickness at each radial distance with outlier removal."""
        profile_means = []
        profile_stds = []

        for i in range(len(radial_bins) - 1):
            r_inner, r_outer = radial_bins[i], radial_bins[i + 1]
            mask = (radius_map >= r_inner) & (radius_map < r_outer)

            if np.any(mask):
                values = thickness_map[mask]
                # Remove outliers
                cleaned_values = self._remove_outliers(values)
                profile_means.append(np.mean(cleaned_values))
                profile_stds.append(np.std(cleaned_values))
            else:
                profile_means.append(np.nan)
                profile_stds.append(np.nan)

        return np.array(profile_means)

    def _calculate_classical_rui(self, radial_profile: np.ndarray) -> float:
        """Calculate classical RUI as coefficient of variation."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 2:
            return 0.0

        mean_thickness = np.mean(valid_profile)
        std_thickness = np.std(valid_profile)

        if mean_thickness == 0:
            return 0.0

        # RUI = 1 - CV (higher values indicate better uniformity)
        cv = std_thickness / mean_thickness
        return max(0.0, 1.0 - cv)

    def _calculate_angular_rui(self, thickness_map: np.ndarray,
                             center: Tuple[int, int],
                             radial_bins: np.ndarray) -> float:
        """Calculate uniformity in angular direction."""
        cy, cx = center
        angular_variations = []

        # Sample at different radii
        for r in radial_bins[1:-1:5]:  # Sample every 5th radius
            angles = np.linspace(0, 2*np.pi, int(360/self.config.angle_resolution_degrees))
            angular_profile = []

            for angle in angles:
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))

                if (0 <= y < thickness_map.shape[0] and
                    0 <= x < thickness_map.shape[1]):
                    angular_profile.append(thickness_map[y, x])

            if len(angular_profile) > 3:
                angular_variations.append(np.std(angular_profile) / np.mean(angular_profile))

        if not angular_variations:
            return 0.0

        mean_angular_cv = np.mean(angular_variations)
        return max(0.0, 1.0 - mean_angular_cv)

    def _calculate_gradient_rui(self, radial_profile: np.ndarray, radial_bins: np.ndarray) -> float:
        """Calculate spatial derivative-based measure."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 2:
            return 0.0

        gradients = np.abs(np.gradient(valid_profile))
        return 1.0 - np.mean(gradients)

    def _calculate_profile_statistics(self, radial_profile: np.ndarray) -> Dict[str, float]:
        """Calculate statistical properties of the radial profile."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 2:
            return {}

        return {
            'profile_mean': np.mean(valid_profile),
            'profile_std': np.std(valid_profile),
            'profile_skewness': stats.skew(valid_profile),
            'profile_kurtosis': stats.kurtosis(valid_profile)
        }

    def _calculate_monotonicity(self, radial_profile: np.ndarray) -> float:
        """Calculate the monotonicity of the radial profile."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 2:
            return 0.0

        # Using Spearman's rank correlation to measure monotonicity
        spearman_corr, _ = stats.spearmanr(np.arange(len(valid_profile)), valid_profile)
        return spearman_corr

    def _calculate_edge_uniformity(self, radial_profile: np.ndarray) -> float:
        """Calculate the uniformity at the edge of the mat."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 10:
            return 0.0

        edge_profile = valid_profile[-10:]
        mean_edge = np.mean(edge_profile)
        if mean_edge == 0:
            return 0.0
        return 1.0 - (np.std(edge_profile) / mean_edge)

    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove outliers from a 1D array of values."""
        if len(values) < 3:
            return values

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return values

        return values[np.abs(values - mean) < self.config.outlier_threshold_std * std]
