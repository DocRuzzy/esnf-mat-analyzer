import numpy as np
from typing import Dict

class UncertaintyQuantifier:
    """Quantify measurement uncertainties for publication-quality error reporting."""

    def quantify_metric_uncertainty(self, metric_values: np.ndarray,
                                  measurement_noise: float,
                                  systematic_error: float = 0.0
                                 ) -> Dict[str, float]:
        """Calculate comprehensive uncertainty estimates.

        Args:
            metric_values: Array of metric measurements
            measurement_noise: Estimated measurement noise level
            systematic_error: Known systematic error contribution

        Returns:
            Uncertainty quantification results
        """
        n_measurements = len(metric_values)

        # Statistical uncertainty
        statistical_uncertainty = np.std(metric_values) / np.sqrt(n_measurements)

        # Combined uncertainty (Type A + Type B)
        combined_uncertainty = np.sqrt(
            statistical_uncertainty**2 +
            measurement_noise**2 +
            systematic_error**2
        )

        # Confidence intervals
        confidence_95 = 1.96 * statistical_uncertainty

        return {
            'statistical_uncertainty': statistical_uncertainty,
            'measurement_noise': measurement_noise,
            'systematic_error': systematic_error,
            'combined_uncertainty': combined_uncertainty,
            'confidence_interval_95': confidence_95,
            'relative_uncertainty_percent': (combined_uncertainty / np.mean(metric_values)) * 100
        }
