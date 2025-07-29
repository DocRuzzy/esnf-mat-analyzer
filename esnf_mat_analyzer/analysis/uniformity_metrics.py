"""
Uniformity metrics for nanofiber thickness analysis.

This module implements various metrics for quantifying the uniformity
of nanofiber thickness, including:
- Radial Uniformity Index
- Gini Coefficient
- Thickness Range Ratio
"""

import numpy as np
import cv2
from typing import Tuple, List
import logging

from esnf_mat_analyzer.core.interfaces import UniformityMetricInterface
from esnf_mat_analyzer.core.data_types import UniformityConfig

class RadialUniformityIndex(UniformityMetricInterface):
    """
    Calculates uniformity by measuring variation along radial lines from the center.
    
    A higher value indicates better uniformity.
    """
    
    def __init__(self, config: UniformityConfig):
        """
        Initialize the Radial Uniformity Index calculator.
        
        Args:
            config: Configuration parameters for uniformity analysis
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "Radial Uniformity Index"
    
    @property
    def description(self) -> str:
        return ("Measures thickness variation along radial lines from the center. "
                "Values closer to 1.0 indicate better uniformity.")
    
    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate the Radial Uniformity Index.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            contour: The contour of the shape.
            
        Returns:
            Radial Uniformity Index (0-1, higher is more uniform)
        """
        self.logger.debug("Calculating Radial Uniformity Index")

        if contour is None:
            self.logger.warning("Contour is None, cannot calculate Radial Uniformity Index.")
            return 0.0

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            self.logger.warning("Contour has zero area, cannot calculate centroid.")
            return 0.0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Extract dimensions
        height, width = thickness_map.shape
        
        # Create angles for radial lines
        angles = np.linspace(0, 2*np.pi, self.config.num_radial_lines, endpoint=False)
        
        # Extract radial profiles
        radial_profiles = []
        for angle in angles:
            # Calculate points along the radial line
            # Use a large number of points to ensure we capture enough detail
            max_radius = np.sqrt(width**2 + height**2)
            distances = np.arange(0, max_radius, 1.0)
            
            # Calculate coordinates along the radial line
            x_coords = cx + np.cos(angle) * distances
            y_coords = cy + np.sin(angle) * distances
            
            # Filter out points outside the image
            valid_indices = (
                (x_coords >= 0) & 
                (x_coords < width) & 
                (y_coords >= 0) & 
                (y_coords < height)
            )
            x_coords = x_coords[valid_indices].astype(int)
            y_coords = y_coords[valid_indices].astype(int)
            
            # Extract thickness values along the line
            thickness_values = thickness_map[y_coords, x_coords]
            
            # Filter points based on mask
            mask_values = mask[y_coords, x_coords]
            valid_thickness = thickness_values[mask_values > 0]
            
            if len(valid_thickness) > 0:
                # Apply smoothing if configured
                if self.config.smoothing_factor > 0:
                    kernel_size = max(3, int(len(valid_thickness) * self.config.smoothing_factor / 10))
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Ensure odd kernel size
                    valid_thickness = np.convolve(
                        valid_thickness, 
                        np.ones(kernel_size)/kernel_size, 
                        mode='valid'
                    )
                
                radial_profiles.append(valid_thickness)
        
        # Remove empty profiles
        radial_profiles = [p for p in radial_profiles if len(p) > 0]
        
        if not radial_profiles:
            self.logger.warning("No valid radial profiles found")
            return 0.0
        
        # Calculate coefficient of variation for each profile
        profile_cvs = []
        for profile in radial_profiles:
            if len(profile) < 2:
                continue
                
            mean = np.mean(profile)
            std = np.std(profile)
            
            # Avoid division by zero
            if mean > 0:
                cv = std / mean
                profile_cvs.append(cv)
        
        if not profile_cvs:
            self.logger.warning("Could not calculate coefficient of variation for any profile")
            return 0.0
        
        # Calculate average CV across all profiles
        mean_cv = np.mean(profile_cvs)
        
        # Convert to uniformity index (higher is better)
        rui = 1.0 - min(mean_cv, 1.0)  # Cap at 1.0 for extreme cases
        
        return rui


class GiniCoefficient(UniformityMetricInterface):
    """
    Calculates the Gini coefficient as a measure of thickness inequality.
    
    Lower values indicate better uniformity (more equal thickness distribution).
    """
    
    @property
    def name(self) -> str:
        return "Gini Coefficient"
    
    @property
    def description(self) -> str:
        return ("Measures statistical dispersion as thickness inequality. "
                "Values closer to 0 indicate better uniformity (more equal distribution).")
    
    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate the Gini Coefficient.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            contour: The contour of the shape (not used for this calculation).
            
        Returns:
            Gini coefficient (0-1, lower is more uniform)
        """
        # Extract thickness values within the mask
        valid_thickness = thickness_map[mask > 0]
        
        if len(valid_thickness) <= 1:
            return 0.0  # Perfect equality with 0 or 1 value
        
        # Sort values
        sorted_values = np.sort(valid_thickness)
        n = len(sorted_values)
        
        # Calculate cumulative sum and total sum
        cumsum = np.cumsum(sorted_values)
        total_sum = cumsum[-1]
        
        # Check for zero-sum case
        if total_sum == 0:
            return 0.0  # All values are zero
        
        # Calculate Lorenz curve coordinates
        # x = cumulative share of population (equally spaced)
        # y = cumulative share of thickness
        x = np.arange(1, n + 1) / n
        y = cumsum / total_sum
        
        # Calculate Gini coefficient as 1 - 2 * area under Lorenz curve
        # Use trapezoidal integration to calculate area
        area = np.trapz(y, x=x)
        gini = 1 - 2 * area
        
        return gini


class ThicknessRangeRatio(UniformityMetricInterface):
    """
    Calculates the ratio of minimum to maximum thickness.
    
    Higher values indicate better uniformity.
    """
    
    @property
    def name(self) -> str:
        return "Thickness Range Ratio"
    
    @property
    def description(self) -> str:
        return ("Ratio of minimum to maximum thickness. "
                "Values closer to 1.0 indicate better uniformity.")
    
    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate the Thickness Range Ratio (min/max).
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            contour: The contour of the shape (not used for this calculation).
            
        Returns:
            Thickness Range Ratio (0-1, higher is more uniform)
        """
        # Extract thickness values within the mask
        valid_thickness = thickness_map[mask > 0]
        
        if len(valid_thickness) == 0:
            return 0.0
        
        # Get minimum and maximum thickness
        min_thickness = np.min(valid_thickness)
        max_thickness = np.max(valid_thickness)
        
        # Calculate ratio (avoid division by zero)
        if max_thickness > 0:
            ratio = min_thickness / max_thickness
        else:
            # All values are zero
            ratio = 1.0
        
        return ratio
