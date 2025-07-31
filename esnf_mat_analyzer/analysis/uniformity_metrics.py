
"""
Uniformity metrics for ESNF Mat Analyzer.

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import numpy as np
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
        roi_thickness = roi_thickness[roi_thickness > 0]
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
