import numpy as np
from typing import Dict
import logging


class MultiScaleUniformityAnalyzer:
    """Multi-scale uniformity analysis using wavelet decomposition.

    This analyzer evaluates nanofiber mat uniformity across multiple spatial scales
    using wavelet decomposition. It provides detailed insights into:
    
    Scale Levels:
    - Scale 0: Large-scale thickness gradients and overall uniformity
    - Scale 1: High-frequency details (individual fiber variations)
    - Scale 2: Medium-high frequency (small fiber bundles)
    - Scale 3: Medium frequency (processing defects, thickness patches)
    - Scale 4: Fine frequency (surface texture)
    
    Metrics Used:
    - Scale 0 (Approximation): Coefficient of Variation (CV) → exp(-CV)
    - Scales 1-4 (Details): Normalized Standard Deviation → 1/(1+norm_std)
    
    The analysis focuses only on the ROI (Region of Interest) to exclude
    rulers, background, and non-mat regions from the evaluation.
    
    Example:
        analyzer = MultiScaleUniformityAnalyzer(wavelet='db4', levels=4, max_coefficients=100000)
        results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
        # Results: {'scale_0_uniformity': 0.85, 'scale_1_uniformity': 0.92, ...}
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 2, max_coefficients: int = 5000):
        self.wavelet = wavelet
        self.levels = levels
        self.max_coefficients = max_coefficients
        self.logger = logging.getLogger(__name__)

    def analyze_multiscale_uniformity(self, thickness_map: np.ndarray,
                                    roi_mask: np.ndarray,
                                    max_coefficients: int = None
                                   ) -> Dict[str, np.ndarray]:
        """Perform multi-scale uniformity analysis using wavelet decomposition.

        This method decomposes the thickness map into multiple scales using wavelets,
        then evaluates uniformity at each scale within the specified ROI.

        Process:
        1. Apply wavelet decomposition to separate scales
        2. Downsample ROI mask to match each scale's resolution
        3. Extract coefficients within ROI for each scale
        4. Calculate scale-specific uniformity metrics:
           - Scale 0: CV-based uniformity (large-scale trends)
           - Scales 1-4: Range-normalized std uniformity (texture/details)
        5. Return uniformity scores for each valid scale

        Args:
            thickness_map: 2D thickness distribution (full image)
            roi_mask: Boolean mask defining analysis region (excludes ruler, background)

        Returns:
            Dictionary with scale-dependent uniformity metrics:
            {
                'scale_0_uniformity': float,  # Large-scale uniformity (0-1)
                'scale_1_uniformity': float,  # Fine detail uniformity (0-1)
                'scale_2_uniformity': float,  # Medium-fine uniformity (0-1)
                'scale_3_uniformity': float,  # Medium uniformity (0-1)
                'scale_4_uniformity': float   # Coarse detail uniformity (0-1)
            }
            
        Notes:
            - Higher scores (closer to 1.0) indicate better uniformity
            - Scale 0 captures overall thickness consistency
            - Scales 1-4 capture texture and local variation consistency
            - Only ROI region is analyzed; ruler and background are excluded
            - Falls back to multi-resolution analysis if wavelets unavailable
        """
        try:
            import pywt
        except ImportError:
            # Fallback to simple multi-resolution analysis
            return self._fallback_multiscale_analysis(thickness_map, roi_mask)

        # Use instance default if not provided
        if max_coefficients is None:
            max_coefficients = self.max_coefficients

        # Validate inputs
        if thickness_map.size == 0 or roi_mask.size == 0:
            return {'scale_0_uniformity': 0.0}
        
        if not np.any(roi_mask):
            # No valid ROI region
            self.logger.warning("ROI mask has no True values - no valid analysis region")
            return {'scale_0_uniformity': 0.0}

        # Debug information
        roi_true_pixels = np.sum(roi_mask)
        roi_percentage = (roi_true_pixels / roi_mask.size) * 100
        thickness_range = (np.min(thickness_map), np.max(thickness_map))
        valid_thickness_in_roi = thickness_map[roi_mask]
        if len(valid_thickness_in_roi) > 0:
            roi_thickness_range = (np.min(valid_thickness_in_roi), np.max(valid_thickness_in_roi))
        else:
            roi_thickness_range = (0, 0)
            
        self.logger.debug(f"Multiscale analysis: ROI covers {roi_percentage:.1f}% of image ({roi_true_pixels} pixels)")
        self.logger.debug(f"Thickness map range: {thickness_range}, ROI thickness range: {roi_thickness_range}")

        # Don't apply NaN before wavelet decomposition as it corrupts coefficients
        # Instead, we'll apply ROI masking to each scale separately
        
        # Wavelet decomposition on original thickness map
        coeffs = pywt.wavedec2(thickness_map, self.wavelet, level=self.levels)

        scale_uniformity = {}
        total_valid_scales = 0
        
        # Process each scale level
        for level in range(self.levels + 1):  # Include level 0 through self.levels
            scale_data = None
            
            if level == 0:
                # Approximation coefficients (lowest frequency, largest scale)
                scale_data = coeffs[0]
                
                # Apply ROI mask by downsampling it to match coefficient size
                if scale_data.shape != roi_mask.shape:
                    # Downsample ROI mask to match coefficient dimensions
                    downsample_factor_y = roi_mask.shape[0] // scale_data.shape[0]
                    downsample_factor_x = roi_mask.shape[1] // scale_data.shape[1]
                    if downsample_factor_y > 0 and downsample_factor_x > 0:
                        downsampled_mask = roi_mask[::downsample_factor_y, ::downsample_factor_x]
                        # Ensure exact dimensions match
                        if downsampled_mask.shape != scale_data.shape:
                            downsampled_mask = downsampled_mask[:scale_data.shape[0], :scale_data.shape[1]]
                    else:
                        downsampled_mask = roi_mask
                else:
                    downsampled_mask = roi_mask
                    
                # Apply mask to scale data
                if downsampled_mask.shape == scale_data.shape:
                    scale_data = scale_data[downsampled_mask]
                else:
                    scale_data = scale_data.flatten()
                    
            elif level <= len(coeffs) - 1:
                # Detail coefficients (higher frequency, smaller scales)
                detail_coeffs = coeffs[level]
                
                if isinstance(detail_coeffs, tuple) and len(detail_coeffs) == 3:
                    # Extract horizontal, vertical, and diagonal detail coefficients
                    cH, cV, cD = detail_coeffs
                    
                    # Apply ROI masking to each detail coefficient array
                    masked_details = []
                    for coeff_array in [cH, cV, cD]:
                        if coeff_array.shape != roi_mask.shape:
                            # Downsample ROI mask to match coefficient dimensions
                            downsample_factor_y = roi_mask.shape[0] // coeff_array.shape[0]
                            downsample_factor_x = roi_mask.shape[1] // coeff_array.shape[1]
                            if downsample_factor_y > 0 and downsample_factor_x > 0:
                                downsampled_mask = roi_mask[::downsample_factor_y, ::downsample_factor_x]
                                # Ensure exact dimensions match
                                if downsampled_mask.shape != coeff_array.shape:
                                    downsampled_mask = downsampled_mask[:coeff_array.shape[0], :coeff_array.shape[1]]
                            else:
                                downsampled_mask = roi_mask
                        else:
                            downsampled_mask = roi_mask
                            
                        # Apply mask and extract valid coefficients
                        if downsampled_mask.shape == coeff_array.shape:
                            masked_coeffs = coeff_array[downsampled_mask]
                        else:
                            masked_coeffs = coeff_array.flatten()
                            
                        if len(masked_coeffs) > 0:
                            masked_details.append(masked_coeffs)
                    
                    # Combine all detail coefficients for this scale
                    if masked_details:
                        scale_data = np.concatenate(masked_details)
                    else:
                        continue
                else:
                    continue
            else:
                continue

            # Validate scale data
            if scale_data is None or len(scale_data) == 0:
                continue  # Don't add 0.0 uniformity for invalid scales

            # Ensure scale_data is 1D and remove any remaining NaN values
            if scale_data.ndim > 1:
                scale_data = scale_data.flatten()

            # Remove NaN and infinite values
            valid_data = scale_data[np.isfinite(scale_data)]

            # Subsample if too large
            if len(valid_data) > max_coefficients:
                self.logger.warning(f"Subsampling {len(valid_data)} coefficients to {max_coefficients} for scale {level} to avoid memory issues.")
                indices = np.random.choice(len(valid_data), max_coefficients, replace=False)
                valid_data = valid_data[indices]

            if len(valid_data) < 2:  # Need at least 2 values for meaningful statistics
                continue  # Don't add 0.0 uniformity for scales with insufficient data

            # Calculate uniformity using appropriate metric for each scale
            std_val = np.std(valid_data)
            mean_val = np.mean(valid_data)

            self.logger.debug(f"Scale {level}: {len(valid_data)} valid data points, mean={mean_val:.6f}, std={std_val:.6f}")

            if level == 0:
                # For approximation coefficients (scale 0), use coefficient of variation
                # Scale 0 represents large-scale thickness trends and gradients
                # CV is appropriate here because approximation coefficients have meaningful means
                if abs(mean_val) > 1e-12:
                    cv = std_val / abs(mean_val)
                    # Convert to uniformity score using exponential decay
                    # exp(-CV) gives values close to 1 for low CV (uniform) and close to 0 for high CV
                    uniformity_value = np.exp(-cv)
                    self.logger.debug(f"Scale {level}: CV={cv:.6f}, uniformity={uniformity_value:.6f}")
                else:
                    # If mean is effectively zero, use std-based measure
                    if std_val < 1e-12:
                        uniformity_value = 1.0
                        self.logger.debug(f"Scale {level}: Zero variance case, uniformity=1.0")
                    else:
                        uniformity_value = 1.0 / (1.0 + std_val)
                        self.logger.debug(f"Scale {level}: Zero mean case, std={std_val:.6f}, uniformity={uniformity_value:.6f}")
            else:
                # For detail coefficients (scales 1-4), use normalized standard deviation
                # Detail coefficients represent texture variations, edges, and local features
                # They often have near-zero means, making CV unstable
                # Use range-normalized std to measure variation relative to coefficient range
                data_range = np.max(valid_data) - np.min(valid_data)
                if data_range > 1e-12:
                    # Normalized standard deviation: std relative to data range
                    normalized_std = std_val / data_range
                    # Convert to uniformity score: lower variation = higher uniformity
                    uniformity_value = 1.0 / (1.0 + normalized_std)
                    self.logger.debug(f"Scale {level}: range={data_range:.6f}, norm_std={normalized_std:.6f}, uniformity={uniformity_value:.6f}")
                else:
                    # All values are essentially identical - perfect uniformity
                    uniformity_value = 1.0
                    self.logger.debug(f"Scale {level}: Zero range case, uniformity=1.0")
            
            # Ensure uniformity is in valid range
            scale_uniformity[f'scale_{level}_uniformity'] = np.clip(uniformity_value, 0.0, 1.0)
            total_valid_scales += 1

        # Ensure we have at least one scale measurement
        if not scale_uniformity:
            self.logger.warning("No valid scales found in multiscale analysis")
            scale_uniformity['scale_0_uniformity'] = 0.0
        else:
            self.logger.debug(f"Multiscale analysis completed with {len(scale_uniformity)} valid scales: {list(scale_uniformity.keys())}")

        return scale_uniformity

    def create_multiscale_heatmap(self, thickness_map: np.ndarray, 
                                 roi_mask: np.ndarray, 
                                 window_size: int = 64) -> Dict[str, np.ndarray]:
        """Create spatial heatmaps of multiscale uniformity.
        
        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            window_size: Size of sliding window for local analysis
            
        Returns:
            Dictionary of spatial uniformity maps for each scale
        """
        try:
            import pywt
        except ImportError:
            self.logger.warning("PyWavelets not available, cannot create multiscale heatmap")
            return {}
            
        if thickness_map.size == 0 or roi_mask.size == 0:
            return {}
            
        height, width = thickness_map.shape
        half_window = window_size // 2
        
        # Initialize output maps for each scale
        scale_maps = {}
        for level in range(self.levels + 1):
            scale_maps[f'scale_{level}_uniformity_map'] = np.zeros((height, width))
        
        # Slide window across the image
        for y in range(half_window, height - half_window, half_window):
            for x in range(half_window, width - half_window, half_window):
                # Extract local window
                y_start, y_end = y - half_window, y + half_window
                x_start, x_end = x - half_window, x + half_window
                
                local_thickness = thickness_map[y_start:y_end, x_start:x_end]
                local_mask = roi_mask[y_start:y_end, x_start:x_end]
                
                # Skip if window doesn't have enough ROI coverage
                if np.sum(local_mask) < (window_size * window_size * 0.3):
                    continue
                    
                # Analyze local multiscale uniformity
                local_results = self.analyze_multiscale_uniformity(local_thickness, local_mask)
                
                # Assign results to output maps
                for scale_key, uniformity_value in local_results.items():
                    map_key = scale_key.replace('uniformity', 'uniformity_map')
                    if map_key in scale_maps:
                        scale_maps[map_key][y_start:y_end, x_start:x_end] = uniformity_value
        
        # Apply ROI mask to all output maps
        for map_key in scale_maps:
            scale_maps[map_key] = np.where(roi_mask, scale_maps[map_key], np.nan)
            
        return scale_maps

    def _fallback_multiscale_analysis(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback multi-resolution analysis if pywt is not installed."""
        scale_uniformity = {}

        # Validate inputs
        if thickness_map.size == 0 or roi_mask.size == 0:
            return {'scale_0_uniformity': 0.0}
        
        if not np.any(roi_mask):
            # No valid ROI region
            self.logger.warning("ROI mask has no True values - no valid analysis region (fallback)")
            return {'scale_0_uniformity': 0.0}

        # Debug information
        roi_true_pixels = np.sum(roi_mask)
        roi_percentage = (roi_true_pixels / roi_mask.size) * 100
        self.logger.debug(f"Fallback multiscale analysis: ROI covers {roi_percentage:.1f}% of image ({roi_true_pixels} pixels)")

        # Include level 0 through self.levels for consistency with wavelet version
        for level in range(self.levels + 1):
            scale_factor = 2**level
            if scale_factor > min(thickness_map.shape):
                # If scale factor is too large, skip this scale
                continue

            # Downsample the image and mask
            if scale_factor == 1:
                # Level 0: use original resolution
                downsampled_map = thickness_map
                downsampled_mask = roi_mask
            else:
                # Higher levels: downsample
                downsampled_map = thickness_map[::scale_factor, ::scale_factor]
                downsampled_mask = roi_mask[::scale_factor, ::scale_factor]

            # Apply mask and extract valid data
            if downsampled_mask.shape == downsampled_map.shape:
                valid_data = downsampled_map[downsampled_mask]
            else:
                # If shapes don't match, use flattened data
                valid_data = downsampled_map.flatten()
            
            # Remove any invalid values
            valid_data = valid_data[np.isfinite(valid_data)]
            
            if len(valid_data) < 2:  # Need at least 2 values for meaningful statistics
                continue  # Don't add 0.0 uniformity for scales with insufficient data
                
            # Calculate uniformity using appropriate metric for each scale
            std_val = np.std(valid_data)
            mean_val = np.mean(valid_data)
            
            if level == 0:
                # For level 0 (original resolution), use coefficient of variation
                if abs(mean_val) > 1e-12:
                    cv = std_val / abs(mean_val)
                    # Convert to uniformity score using exponential decay
                    uniformity_value = np.exp(-cv)
                else:
                    # If mean is effectively zero, use std-based measure
                    if std_val < 1e-12:
                        uniformity_value = 1.0
                    else:
                        uniformity_value = 1.0 / (1.0 + std_val)
            else:
                # For higher levels (downsampled), use range-normalized std
                data_range = np.max(valid_data) - np.min(valid_data)
                if data_range > 1e-12:
                    # Normalized standard deviation
                    normalized_std = std_val / data_range
                    # Convert to uniformity score (0-1, where 1 is perfect uniformity)
                    uniformity_value = 1.0 / (1.0 + normalized_std)
                else:
                    # All values are essentially identical
                    uniformity_value = 1.0
            
            # Ensure uniformity is in valid range
            scale_uniformity[f'scale_{level}_uniformity'] = np.clip(uniformity_value, 0.0, 1.0)

        # Ensure we have at least one scale measurement
        if not scale_uniformity:
            self.logger.warning("No valid scales found in fallback multiscale analysis")
            scale_uniformity['scale_0_uniformity'] = 0.0
        else:
            self.logger.debug(f"Fallback multiscale analysis completed with {len(scale_uniformity)} valid scales: {list(scale_uniformity.keys())}")

        return scale_uniformity
