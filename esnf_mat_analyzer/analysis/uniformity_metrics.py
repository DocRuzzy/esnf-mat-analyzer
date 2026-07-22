
"""
Uniformity metrics for ESNF Mat Analyzer.

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import logging

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from esnf_mat_analyzer.core.interfaces import IUniformityMetric
from esnf_mat_analyzer.core.data_types import UniformityConfig

class RadialUniformityIndex(IUniformityMetric):
    """
    Calculates uniformity by measuring variation along radial lines from the center.
    
    A higher value indicates better uniformity.
    """
    
    def __init__(self, config: UniformityConfig):
        """
        Initialize the RadialUniformityIndex with configuration.
        
        Args:
            config: Configuration parameters for uniformity analysis
        """
        self.config = config
        self.name = "RadialUniformityIndex"
    
    def calculate_metric(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> float:
        """
        Calculate the Radial Uniformity Index.
        
        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            
        Returns:
            Radial Uniformity Index (0-1, higher is more uniform)
        """
        # Extract ROI data
        roi_indices = np.where(roi_mask)
        if len(roi_indices[0]) == 0:
            return 0.0
        
        roi_thickness = thickness_map[roi_indices]
        
        # Find center of ROI
        center_y = int(np.mean(roi_indices[0]))
        center_x = int(np.mean(roi_indices[1]))
        
        # Calculate radial distances from center
        y_coords, x_coords = roi_indices
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Create radial bins
        max_distance = np.max(distances)
        if max_distance == 0:
            return 1.0  # Perfect uniformity if only one point
        
        num_bins = min(self.config.num_radial_lines, int(max_distance))
        if num_bins < 2:
            return 1.0
            
        bin_edges = np.linspace(0, max_distance, num_bins + 1)
        
        # Calculate mean thickness in each radial bin
        radial_means = []
        for i in range(num_bins):
            bin_mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if np.any(bin_mask):
                bin_thickness = roi_thickness[bin_mask]
                radial_means.append(np.mean(bin_thickness))
        
        if len(radial_means) < 2:
            return 1.0
            
        # Calculate coefficient of variation
        radial_means = np.array(radial_means)
        mean_thickness = np.mean(radial_means)
        
        if mean_thickness == 0:
            return 0.0
            
        cv = np.std(radial_means) / mean_thickness
        
        # Convert to uniformity index (higher is better)
        rui = max(0.0, 1.0 - cv)
        
        return min(1.0, rui)
    
    def calculate(self, thickness_map: np.ndarray, roi_mask: np.ndarray, *args) -> float:
        """
        Legacy method for compatibility with analyzer that expects calculate() method.
        
        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            *args: Additional arguments (ignored for compatibility)
            
        Returns:
            Radial Uniformity Index (0-1, higher is more uniform)
        """
        return self.calculate_metric(thickness_map, roi_mask)

class GiniCoefficient(IUniformityMetric):
    """
    Calculates the Gini coefficient as a measure of thickness inequality.
    
    Lower values indicate better uniformity (more equal thickness distribution).
    """
    
    def __init__(self, config: UniformityConfig = None):
        """
        Initialize the GiniCoefficient with configuration.
        
        Args:
            config: Configuration parameters for uniformity analysis (optional)
        """
        self.config = config
        self.name = "GiniCoefficient"
    
    def calculate_metric(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> float:
        """
        Calculate the Gini Coefficient.
        
        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            
        Returns:
            Gini coefficient (0-1, lower is more uniform)
        """
        # Extract ROI data
        roi_indices = np.where(roi_mask)
        if len(roi_indices[0]) == 0:
            return 1.0  # Maximum inequality for empty ROI
        
        roi_thickness = thickness_map[roi_indices]

        # Remove any non-positive values
        n_total = len(roi_thickness)
        roi_thickness = roi_thickness[roi_thickness > 0]
        n_dropped = n_total - len(roi_thickness)
        if n_dropped > 0.01 * n_total:
            logging.getLogger(__name__).warning(
                f"GiniCoefficient: discarded {n_dropped}/{n_total} non-positive "
                f"thickness values ({100.0 * n_dropped / n_total:.1f}%) — result "
                f"may be biased (check background correction / clipping)"
            )
        if len(roi_thickness) == 0:
            return 1.0
        
        # Sort values
        sorted_thickness = np.sort(roi_thickness)
        n = len(sorted_thickness)
        
        if n == 1:
            return 0.0  # Perfect equality for single value
        
        # Calculate Gini coefficient
        # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        cumsum = np.cumsum(sorted_thickness)
        total_sum = cumsum[-1]
        
        if total_sum == 0:
            return 0.0
        
        # Weighted sum of values
        weighted_sum = np.sum((np.arange(1, n + 1) * sorted_thickness))
        
        gini = (2.0 * weighted_sum) / (n * total_sum) - (n + 1.0) / n
        
        return max(0.0, min(1.0, gini))
    
    def calculate(self, thickness_map: np.ndarray, roi_mask: np.ndarray, *args) -> float:
        """
        Legacy method for compatibility with analyzer that expects calculate() method.
        
        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            *args: Additional arguments (ignored for compatibility)
            
        Returns:
            Gini coefficient (0-1, lower is more uniform)
        """
        return self.calculate_metric(thickness_map, roi_mask)

class ThicknessRangeRatio(IUniformityMetric):
    """
    Calculates the ratio of minimum to maximum thickness.
    
    Higher values indicate better uniformity.
    """
    
    def __init__(self, config: UniformityConfig = None):
        """
        Initialize the ThicknessRangeRatio with configuration.
        
        Args:
            config: Configuration parameters for uniformity analysis (optional)
        """
        self.config = config
        self.name = "ThicknessRangeRatio"
    
    def calculate_metric(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> float:
        """
        Calculate the Thickness Range Ratio (min/max).
        
        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            
        Returns:
            Thickness Range Ratio (0-1, higher is more uniform)
        """
        # Extract ROI data
        roi_indices = np.where(roi_mask)
        if len(roi_indices[0]) == 0:
            return 0.0  # No uniformity for empty ROI
        
        roi_thickness = thickness_map[roi_indices]
        
        # Remove any non-positive values
        roi_thickness = roi_thickness[roi_thickness > 0]
        if len(roi_thickness) == 0:
            return 0.0
        
        min_thickness = np.min(roi_thickness)
        max_thickness = np.max(roi_thickness)
        
        if max_thickness == 0:
            return 0.0
        
        if min_thickness == max_thickness:
            return 1.0  # Perfect uniformity
        
        # Calculate ratio (min/max)
        ratio = min_thickness / max_thickness
        
        return max(0.0, min(1.0, ratio))
    
    def calculate(self, thickness_map: np.ndarray, roi_mask: np.ndarray, *args) -> float:
        """
        Legacy method for compatibility with analyzer that expects calculate() method.

        Args:
            thickness_map: 2D thickness map
            roi_mask: Boolean mask indicating the region of interest
            *args: Additional arguments (ignored for compatibility)

        Returns:
            Thickness Range Ratio (0-1, higher is more uniform)
        """
        return self.calculate_metric(thickness_map, roi_mask)


class SpatialCVMap:
    """
    Computes a 2D spatial coefficient of variation map using a sliding window.

    Each pixel in the output map represents the local CV (std/mean) computed
    over a surrounding window, indicating WHERE non-uniformity exists spatially.
    """

    def __init__(self, window_size: int = 51):
        self.window_size = window_size
        self.name = "SpatialCVMap"

    def compute(
        self, thickness_map: np.ndarray, roi_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute the spatial CV map.

        Args:
            thickness_map: 2D intensity/thickness map
            roi_mask: Boolean mask of the region of interest

        Returns:
            Tuple of (local_cv_map, summary_stats_dict).
            local_cv_map has the same shape as thickness_map.
            summary_stats_dict contains 'mean_local_cv', 'max_local_cv', 'std_local_cv'.
        """
        bool_mask = roi_mask.astype(bool)
        data = thickness_map.astype(np.float32)
        ws = self.window_size

        valid_float = bool_mask.astype(np.float32)
        count_map = cv2.boxFilter(valid_float, -1, (ws, ws), normalize=False)
        count_map = np.maximum(count_map, 1)

        data_masked = np.where(bool_mask, data, 0).astype(np.float32)
        local_sum = cv2.boxFilter(data_masked, -1, (ws, ws), normalize=False)
        local_sum_sq = cv2.boxFilter(data_masked ** 2, -1, (ws, ws), normalize=False)

        local_mean = local_sum / count_map
        local_var = np.maximum((local_sum_sq / count_map) - (local_mean ** 2), 0)
        local_std = np.sqrt(local_var)
        local_cv = np.where(local_mean > 1e-6, local_std / local_mean, 0)

        # Zero out non-ROI pixels
        local_cv[~bool_mask] = 0

        valid_cv = local_cv[bool_mask]
        stats = {}
        if len(valid_cv) > 0:
            stats['mean_local_cv'] = float(np.mean(valid_cv))
            stats['max_local_cv'] = float(np.max(valid_cv))
            stats['std_local_cv'] = float(np.std(valid_cv))
        else:
            stats['mean_local_cv'] = 0.0
            stats['max_local_cv'] = 0.0
            stats['std_local_cv'] = 0.0

        return local_cv, stats


class SectorUniformity:
    """
    Divides the mat into angular sectors from its centroid and computes
    per-sector intensity statistics, revealing directional non-uniformity.
    """

    def __init__(self, num_sectors: int = 8):
        self.num_sectors = num_sectors
        self.name = "SectorUniformity"

    def compute(
        self, thickness_map: np.ndarray, roi_mask: np.ndarray
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Compute per-sector statistics.

        Args:
            thickness_map: 2D intensity/thickness map
            roi_mask: Boolean mask of the region of interest

        Returns:
            Tuple of (sector_list, summary_dict).
            sector_list: List of dicts with keys 'sector', 'angle_deg', 'mean', 'std', 'cv', 'pixel_count'.
            summary_dict: Contains 'sector_cv_range' (max sector CV - min sector CV)
                          and 'max_sector_cv', 'min_sector_cv'.
        """
        bool_mask = roi_mask.astype(bool)
        data = thickness_map.astype(np.float32)

        ys, xs = np.where(bool_mask)
        if len(ys) == 0:
            return [], {'sector_cv_range': 0.0, 'max_sector_cv': 0.0, 'min_sector_cv': 0.0}

        mat_values = data[bool_mask]
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        angles = np.arctan2(ys - cy, xs - cx)  # -pi to pi

        sector_width = 2 * np.pi / self.num_sectors
        sectors = []

        for i in range(self.num_sectors):
            angle_min = -np.pi + i * sector_width
            angle_max = angle_min + sector_width
            sector_mask = (angles >= angle_min) & (angles < angle_max)
            sector_vals = mat_values[sector_mask]

            if len(sector_vals) > 10:
                s_mean = float(np.mean(sector_vals))
                s_std = float(np.std(sector_vals))
                s_cv = s_std / s_mean if s_mean > 0 else 0
                sectors.append({
                    'sector': i,
                    'angle_deg': float((angle_min + angle_max) / 2 * 180 / np.pi),
                    'mean': s_mean,
                    'std': s_std,
                    'cv': s_cv,
                    'pixel_count': int(len(sector_vals)),
                })

        cvs = [s['cv'] for s in sectors]
        summary = {
            'sector_cv_range': (max(cvs) - min(cvs)) if cvs else 0.0,
            'max_sector_cv': max(cvs) if cvs else 0.0,
            'min_sector_cv': min(cvs) if cvs else 0.0,
        }

        return sectors, summary
