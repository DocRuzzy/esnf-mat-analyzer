import numpy as np
from typing import Dict

class MultiScaleUniformityAnalyzer:
    """Multi-scale uniformity analysis using wavelet decomposition.

    Analyzes uniformity at different length scales to identify:
    - Fine-scale texture variations
    - Medium-scale defects
    - Large-scale thickness gradients
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels

    def analyze_multiscale_uniformity(self, thickness_map: np.ndarray,
                                    roi_mask: np.ndarray
                                   ) -> Dict[str, np.ndarray]:
        """Perform multi-scale uniformity analysis.

        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region

        Returns:
            Scale-dependent uniformity metrics
        """
        try:
            import pywt
        except ImportError:
            # Fallback to simple multi-resolution analysis
            return self._fallback_multiscale_analysis(thickness_map, roi_mask)

        # Apply ROI mask
        masked_thickness = thickness_map.copy()
        masked_thickness[~roi_mask] = np.nan

        # Wavelet decomposition
        coeffs = pywt.wavedec2(masked_thickness, self.wavelet, level=self.levels)

        scale_uniformity = {}
        for level in range(self.levels):
            # Extract coefficients at this scale
            if level == 0:
                scale_data = coeffs[0]  # Approximation coefficients
            else:
                scale_data = coeffs[level]  # Detail coefficients

            # Calculate uniformity at this scale
            valid_data = scale_data[~np.isnan(scale_data)]
            if len(valid_data) > 0:
                scale_uniformity[f'scale_{level}_uniformity'] = 1.0 - (
                    np.std(valid_data) / np.mean(np.abs(valid_data))
                )

        return scale_uniformity

    def _fallback_multiscale_analysis(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback multi-resolution analysis if pywt is not installed."""
        scale_uniformity = {}

        for level in range(self.levels):
            scale_factor = 2**level
            if scale_factor > min(thickness_map.shape):
                break

            # Downsample the image
            downsampled_map = thickness_map[::scale_factor, ::scale_factor]
            downsampled_mask = roi_mask[::scale_factor, ::scale_factor]

            # Calculate uniformity at this scale
            valid_data = downsampled_map[downsampled_mask]
            if len(valid_data) > 0:
                scale_uniformity[f'scale_{level}_uniformity'] = 1.0 - (
                    np.std(valid_data) / np.mean(np.abs(valid_data))
                )

        return scale_uniformity
