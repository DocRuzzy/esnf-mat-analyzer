"""
Mat-scale anisotropy analysis using FFT for electrospun fiber mats.
Analyzes bulk directional preference in fiber deposition, not individual fibers.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class MatAnisotropyAnalyzer:
    """
    Analyze anisotropy in electrospun fiber mats using FFT-based methods.
    
    This analyzes the bulk directional preference of fiber deposition
    based on regional density variations, not individual fiber orientation.
    """
    
    def __init__(self):
        pass
    
    def analyze_mat_anisotropy(self, thickness_map: np.ndarray, 
                              mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze bulk anisotropy in fiber mat using FFT.
        
        Args:
            thickness_map: 2D thickness/density map of the mat
            mask: Optional mask for ROI analysis
            
        Returns:
            Dictionary with anisotropy metrics
        """
        if mask is not None:
            # Apply mask
            masked_thickness = thickness_map * mask
        else:
            masked_thickness = thickness_map
            
        # Remove mean to focus on variations
        thickness_centered = masked_thickness - np.mean(masked_thickness)
        
        # Apply window function to reduce edge effects
        windowed = self._apply_hanning_window(thickness_centered)
        
        # 2D FFT
        fft = np.fft.fft2(windowed)
        power_spectrum = np.abs(fft) ** 2
        
        # Shift to center
        power_spectrum_shifted = np.fft.fftshift(power_spectrum)
        
        # Calculate anisotropy metrics
        anisotropy_index = self._calculate_anisotropy_index(power_spectrum_shifted)
        preferred_angle = self._calculate_preferred_angle(power_spectrum_shifted)
        directional_variance = self._calculate_directional_variance(power_spectrum_shifted)
        
        return {
            'anisotropy_index': anisotropy_index,
            'preferred_angle_degrees': preferred_angle,
            'directional_variance': directional_variance,
            'orientation_strength': self._calculate_orientation_strength(power_spectrum_shifted)
        }
    
    def _apply_hanning_window(self, image: np.ndarray) -> np.ndarray:
        """Apply 2D Hanning window to reduce edge effects."""
        rows, cols = image.shape
        hann_row = np.hanning(rows).reshape(-1, 1)
        hann_col = np.hanning(cols).reshape(1, -1)
        window = hann_row @ hann_col
        return image * window

    def _safe_weighted_average(self, values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> float:
        """Compute a weighted average safely.

        Returns `default` if weights sum to zero or are all non-finite. This
        prevents ZeroDivisionError in np.average when running on uniform or
        empty spectra. The function also handles NaNs by ignoring them.
        """
        try:
            vals = np.asarray(values, dtype=float)
            w = np.asarray(weights, dtype=float)
        except Exception:
            return default

        # Mask invalid entries
        valid = np.isfinite(vals) & np.isfinite(w)
        if not np.any(valid):
            return default

        vals = vals[valid]
        w = w[valid]

        wsum = float(np.sum(w))
        if wsum == 0.0 or not np.isfinite(wsum):
            return default

        return float(np.sum(vals * w) / wsum)
    
    def _calculate_anisotropy_index(self, power_spectrum: np.ndarray) -> float:
        """
        Calculate anisotropy index from power spectrum.
        
        Returns value between 0 (isotropic) and 1 (highly anisotropic).
        """
        center = np.array(power_spectrum.shape) // 2
        
        # Create radial and angular grids
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        y = y - center[0]
        x = x - center[1]
        
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Focus on mid-frequency range (avoid DC and high-freq noise)
        max_r = min(center) * 0.8
        min_r = min(center) * 0.1
        valid_mask = (r >= min_r) & (r <= max_r)
        
        if not np.any(valid_mask):
            return 0.0
            
        # Calculate angular power distribution
        angles = theta[valid_mask]
        powers = power_spectrum[valid_mask]
        
        # Bin by angle (0 to π, since power spectrum is symmetric)
        angle_bins = np.linspace(-np.pi, np.pi, 36)  # 5-degree bins
        angle_indices = np.digitize(angles, angle_bins) - 1
        
        # Sum power in each angular bin
        angular_power = np.zeros(len(angle_bins) - 1)
        for i in range(len(angle_bins) - 1):
            mask = angle_indices == i
            if np.any(mask):
                angular_power[i] = np.sum(powers[mask])
        
        # Calculate anisotropy as normalized standard deviation
        if np.sum(angular_power) > 0:
            angular_power = angular_power / np.sum(angular_power)
            anisotropy = np.std(angular_power) / np.mean(angular_power)
            # Normalize to [0, 1] range
            return min(anisotropy / 2.0, 1.0)
        else:
            return 0.0
    
    def _calculate_preferred_angle(self, power_spectrum: np.ndarray) -> float:
        """Calculate the preferred orientation angle in degrees."""
        center = np.array(power_spectrum.shape) // 2
        
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        y = y - center[0]
        x = x - center[1]
        
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Focus on mid-frequency range
        max_r = min(center) * 0.8
        min_r = min(center) * 0.1
        valid_mask = (r >= min_r) & (r <= max_r)
        
        if not np.any(valid_mask):
            return 0.0
            
        # Calculate weighted average angle
        angles = theta[valid_mask]
        powers = power_spectrum[valid_mask]
        
        # Convert to unit vectors and average
        cos_angles = np.cos(2 * angles)  # Factor of 2 for orientation (not direction)
        sin_angles = np.sin(2 * angles)

        avg_cos = self._safe_weighted_average(cos_angles, powers)
        avg_sin = self._safe_weighted_average(sin_angles, powers)
        
        # Convert back to angle
        preferred_angle = 0.5 * np.arctan2(avg_sin, avg_cos)
        
        # Convert to degrees and ensure positive
        angle_degrees = np.degrees(preferred_angle)
        if angle_degrees < 0:
            angle_degrees += 180
            
        return angle_degrees
    
    def _calculate_directional_variance(self, power_spectrum: np.ndarray) -> float:
        """Calculate variance in directional power distribution."""
        center = np.array(power_spectrum.shape) // 2
        
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        y = y - center[0]
        x = x - center[1]
        
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Focus on mid-frequency range
        max_r = min(center) * 0.8
        min_r = min(center) * 0.1
        valid_mask = (r >= min_r) & (r <= max_r)
        
        if not np.any(valid_mask):
            return 0.0
            
        angles = theta[valid_mask]
        powers = power_spectrum[valid_mask]
        
        # Calculate directional variance using circular statistics
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        weighted_cos = self._safe_weighted_average(cos_angles, powers)
        weighted_sin = self._safe_weighted_average(sin_angles, powers)
        
        # Circular variance (1 - resultant length)
        resultant_length = np.sqrt(weighted_cos**2 + weighted_sin**2)
        directional_variance = 1.0 - resultant_length
        
        return directional_variance
    
    def _calculate_orientation_strength(self, power_spectrum: np.ndarray) -> float:
        """Calculate the strength of orientation preference."""
        # Use the magnitude of the main orientation vector
        center = np.array(power_spectrum.shape) // 2
        
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        y = y - center[0]
        x = x - center[1]
        
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        max_r = min(center) * 0.8
        min_r = min(center) * 0.1
        valid_mask = (r >= min_r) & (r <= max_r)
        
        if not np.any(valid_mask):
            return 0.0
            
        angles = theta[valid_mask]
        powers = power_spectrum[valid_mask]
        
        # Calculate orientation tensor components
        cos_2theta = np.cos(2 * angles)
        sin_2theta = np.sin(2 * angles)

        # Use safe weighted averages to avoid division errors
        S_xx = self._safe_weighted_average(cos_2theta, powers)
        S_yy = -S_xx
        S_xy = self._safe_weighted_average(sin_2theta, powers)
        
        # Orientation strength is the largest eigenvalue
        orientation_strength = 0.5 * np.sqrt((S_xx - S_yy)**2 + 4*S_xy**2)
        
        return orientation_strength


class MatTextureAnalyzer:
    """
    GLCM texture analysis at mat scale for uniformity characterization.
    """
    
    def __init__(self):
        pass
        
    def analyze_mat_texture(self, thickness_map: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           distances: list = [1, 2, 4],
                           angles: list = [0, 45, 90, 135]) -> Dict[str, float]:
        """
        Analyze texture features of fiber mat using GLCM.
        
        Args:
            thickness_map: 2D thickness map
            mask: Optional ROI mask
            distances: Pixel distances for GLCM calculation
            angles: Angles in degrees for GLCM calculation
            
        Returns:
            Dictionary with texture metrics
        """
        if mask is not None:
            # Apply mask and crop to ROI
            masked_thickness = thickness_map * mask
            # Find bounding box of mask
            rows, cols = np.where(mask > 0)
            if len(rows) == 0:
                return {}
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            cropped_thickness = masked_thickness[min_row:max_row+1, min_col:max_col+1]
            cropped_mask = mask[min_row:max_row+1, min_col:max_col+1]
        else:
            cropped_thickness = thickness_map
            cropped_mask = None
            
        # Normalize to 8-bit for GLCM
        normalized = self._normalize_for_glcm(cropped_thickness)
        
        # Calculate GLCM features
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Convert angles to radians
            angles_rad = [np.deg2rad(angle) for angle in angles]
            
            # Calculate GLCM
            glcm = graycomatrix(
                normalized,
                distances=distances,
                angles=angles_rad,
                levels=64,  # Reduce levels for better statistics
                symmetric=True,
                normed=True
            )
            
            # Calculate texture properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            return {
                'texture_contrast': float(contrast),
                'texture_dissimilarity': float(dissimilarity),
                'texture_homogeneity': float(homogeneity),
                'texture_energy': float(energy),
                'texture_correlation': float(correlation),
                'texture_uniformity_index': float(homogeneity * energy)  # Combined metric
            }
            
        except ImportError:
            logger.warning("scikit-image not available for GLCM analysis")
            return self._simple_texture_analysis(cropped_thickness)
    
    def _normalize_for_glcm(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to appropriate range for GLCM."""
        # Remove extreme outliers
        p2, p98 = np.percentile(image, [2, 98])
        clipped = np.clip(image, p2, p98)
        
        # Normalize to 0-63 (64 levels)
        if clipped.max() > clipped.min():
            normalized = ((clipped - clipped.min()) / (clipped.max() - clipped.min()) * 63)
        else:
            normalized = np.zeros_like(clipped)
        return normalized.astype(np.uint8)
    
    def _simple_texture_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Simple texture analysis without scikit-image."""
        # Local standard deviation (texture contrast)
        kernel = np.ones((5, 5))
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel/25)
        local_sq_mean = cv2.filter2D(image.astype(np.float32)**2, -1, kernel/25)
        local_variance = local_sq_mean - local_mean**2
        texture_contrast = np.mean(np.sqrt(np.maximum(local_variance, 0)))
        
        # Simple homogeneity (inverse of variance)
        texture_homogeneity = 1.0 / (1.0 + texture_contrast)
        
        return {
            'texture_contrast': float(texture_contrast),
            'texture_homogeneity': float(texture_homogeneity),
            'texture_uniformity_index': float(texture_homogeneity)
        }


class PowerSpectralDensityAnalyzer:
    """
    Power spectral density analysis for detecting periodic patterns in mats.
    """
    
    def __init__(self):
        pass
    
    def analyze_psd(self, thickness_map: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze power spectral density to detect periodic patterns.
        
        Args:
            thickness_map: 2D thickness map
            mask: Optional ROI mask
            
        Returns:
            Dictionary with PSD metrics
        """
        if mask is not None:
            masked_thickness = thickness_map * mask
        else:
            masked_thickness = thickness_map
            
        logger.debug(f"PSD analysis input shape: {masked_thickness.shape}")
        
        # Remove mean and apply window
        windowed = self._apply_hanning_window(masked_thickness - np.mean(masked_thickness))
        logger.debug("Applied Hanning window for PSD.")

        # Calculate 2D FFT
        fft = np.fft.fft2(windowed)
        power_spectrum = np.abs(fft) ** 2
        logger.debug(f"Power spectrum calculated, shape: {power_spectrum.shape}, sum: {np.sum(power_spectrum):.2e}")

        # Calculate radial power spectrum
        radial_psd = self._calculate_radial_psd(power_spectrum)
        logger.debug(f"Radial PSD calculated, length: {len(radial_psd)}")

        # Find dominant frequencies
        dominant_freq = self._find_dominant_frequency(radial_psd)
        
        # Calculate periodicity index
        periodicity_index = self._calculate_periodicity_index(radial_psd)
        
        # Calculate spectral centroid and spread
        spectral_centroid = self._calculate_spectral_centroid(radial_psd)
        spectral_spread = self._calculate_spectral_spread(radial_psd)

        metrics = {
            'psd_dominant_frequency': dominant_freq,
            'psd_periodicity_index': periodicity_index,
            'psd_spectral_centroid': spectral_centroid,
            'psd_spectral_spread': spectral_spread,
            'psd_uniformity': 1.0 / (1.0 + spectral_spread) # Example uniformity metric
        }
        logger.info(f"Calculated PSD metrics: {metrics}")
        return metrics
    
    def _apply_hanning_window(self, image: np.ndarray) -> np.ndarray:
        """Apply 2D Hanning window."""
        rows, cols = image.shape
        hann_row = np.hanning(rows).reshape(-1, 1)
        hann_col = np.hanning(cols).reshape(1, -1)
        window = hann_row @ hann_col
        return image * window
    
    def _calculate_radial_psd(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Calculate radially averaged power spectral density."""
        center = np.array(power_spectrum.shape) // 2
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Create radial bins
        max_r = int(min(center))
        radial_profile = np.zeros(max_r)
        
        for i in range(max_r):
            mask = (r >= i) & (r < i + 1)
            if np.any(mask):
                radial_profile[i] = np.mean(power_spectrum[mask])
                
        return radial_profile
    
    def _find_dominant_frequency(self, radial_psd: np.ndarray) -> float:
        """Find the dominant frequency in the radial PSD."""
        # Exclude DC component
        if len(radial_psd) > 1:
            peak_idx = np.argmax(radial_psd[1:]) + 1
            return float(peak_idx)
        return 0.0
    
    def _calculate_periodicity_index(self, radial_psd: np.ndarray) -> float:
        """Calculate index indicating strength of periodic patterns."""
        if len(radial_psd) <= 1:
            return 0.0
            
        # Ratio of peak power to average power
        peak_power = np.max(radial_psd[1:])  # Exclude DC
        avg_power = np.mean(radial_psd[1:])
        
        if avg_power > 0:
            return float(peak_power / avg_power)
        return 0.0
    
    def _calculate_spectral_centroid(self, radial_psd: np.ndarray) -> float:
        """Calculate spectral centroid (frequency center of mass)."""
        frequencies = np.arange(len(radial_psd))
        if np.sum(radial_psd) > 0:
            return float(np.sum(frequencies * radial_psd) / np.sum(radial_psd))
        return 0.0
    
    def _calculate_spectral_spread(self, radial_psd: np.ndarray) -> float:
        """Calculate spectral spread (frequency bandwidth)."""
        centroid = self._calculate_spectral_centroid(radial_psd)
        frequencies = np.arange(len(radial_psd))
        
        if np.sum(radial_psd) > 0:
            spread = np.sum(((frequencies - centroid) ** 2) * radial_psd) / np.sum(radial_psd)
            return float(np.sqrt(spread))
        return 0.0
