"""
Visualization components for nanofiber thickness analysis.

This module implements various visualization techniques for thickness maps
and uniformity metrics, including heatmaps, radial profiles, and metric plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import Tuple, Dict, Optional, Any, List
import logging

from nanofiber_analyzer.core.interfaces import VisualizerInterface
from nanofiber_analyzer.config.config_manager import VisualizationConfig

class Visualizer(VisualizerInterface):
    """
    Component for creating visualizations of nanofiber thickness and uniformity.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration parameters for visualization
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_thickness_heatmap(self, thickness_map: np.ndarray, mask: np.ndarray, 
                                saturation_mask: Optional[np.ndarray] = None) -> Figure:
        """
        Create a heatmap visualization of the thickness map.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            saturation_mask: Optional mask of saturated pixels
            
        Returns:
            Matplotlib Figure object
        """
        self.logger.debug("Creating thickness heatmap")
        
        # Create masked thickness map (only show values within the mask)
        masked_thickness = np.ma.masked_array(thickness_map, mask=~(mask.astype(bool)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Use specified colormap
        cmap = plt.get_cmap(self.config.colormap)
        
        # Create heatmap
        im = ax.imshow(masked_thickness, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Estimated Thickness')
        
        # Highlight saturated regions if requested
        if saturation_mask is not None and self.config.show_saturated:
            # Create a semi-transparent overlay for saturated regions
            saturation_overlay = np.ma.masked_array(
                np.ones_like(thickness_map),
                mask=~(saturation_mask.astype(bool))
            )
            ax.imshow(
                saturation_overlay,
                alpha=0.3,
                cmap=LinearSegmentedColormap.from_list('saturated', ['red', 'red'])
            )
        
        # Add title and labels
        ax.set_title('Nanofiber Thickness Map')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        return fig
    
    def create_shape_visualization(self, image: np.ndarray, contour: np.ndarray) -> Figure:
        """
        Create a visualization of the detected shape on the original image.

        Args:
            image: The original image.
            contour: The detected contour of the mat.

        Returns:
            Matplotlib Figure object
        """
        self.logger.debug("Creating shape visualization")

        # Create a copy of the image to avoid modifying the original
        vis_image = image.copy()

        # Convert to color if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)

        # Draw contour outline
        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax.set_title('Detected Mat Shape')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        return fig

    def create_radial_profile(self, thickness_map: np.ndarray, mask: np.ndarray, 
                             center: Tuple[int, int]) -> Figure:
        """
        Create a visualization of thickness along radial lines.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            center: Center coordinates (x, y) of the circular region
            
        Returns:
            Matplotlib Figure object
        """
        self.logger.debug("Creating radial profile visualization")
        
        # Extract dimensions and center
        height, width = thickness_map.shape
        cx, cy = center
        
        # Calculate maximum radius
        y_indices, x_indices = np.where(mask > 0)
        distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        max_radius = int(np.ceil(np.max(distances)))
        
        # Create distance bins
        r_bins = np.arange(0, max_radius + 1)
        r_values = (r_bins[:-1] + r_bins[1:]) / 2  # Bin centers
        
        # Create array to store average thickness at each radius
        avg_thickness = np.zeros_like(r_values, dtype=float)
        std_thickness = np.zeros_like(r_values, dtype=float)
        
        # Compute distance map
        y_grid, x_grid = np.ogrid[:height, :width]
        distance_map = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        # For each radius bin, calculate average thickness
        for i, (r_min, r_max) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            # Find pixels within the current radius bin and within the mask
            bin_mask = (distance_map >= r_min) & (distance_map < r_max) & (mask > 0)
            
            # Extract thickness values for these pixels
            bin_thickness = thickness_map[bin_mask]
            
            if len(bin_thickness) > 0:
                avg_thickness[i] = np.mean(bin_thickness)
                std_thickness[i] = np.std(bin_thickness)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Plot average thickness vs radius
        ax.plot(r_values, avg_thickness, 'b-', linewidth=2)
        
        # Add shaded region for standard deviation
        ax.fill_between(
            r_values,
            avg_thickness - std_thickness,
            avg_thickness + std_thickness,
            alpha=0.3,
            color='blue'
        )
        
        # Add title and labels
        ax.set_title('Radial Thickness Profile')
        ax.set_xlabel('Distance from Center (pixels)')
        ax.set_ylabel('Average Thickness')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def create_uniformity_visualization(self, thickness_map: np.ndarray, mask: np.ndarray,
                                      metrics: Dict[str, float]) -> Figure:
        """
        Create a visualization of uniformity metrics.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            metrics: Dictionary of metric names and values
            
        Returns:
            Matplotlib Figure object
        """
        self.logger.debug("Creating uniformity metrics visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.5))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        
        # Create histogram of thickness values (upper plot)
        ax1 = fig.add_subplot(gs[0])
        
        # Extract valid thickness values
        valid_thickness = thickness_map[mask > 0]
        
        # Create histogram
        hist, bins, _ = ax1.hist(
            valid_thickness, 
            bins=50, 
            color='skyblue', 
            edgecolor='black', 
            alpha=0.7
        )
        
        # Add vertical lines for quartiles and median
        quartiles = np.percentile(valid_thickness, [25, 50, 75])
        line_colors = ['green', 'red', 'green']
        line_labels = ['Q1', 'Median', 'Q3']
        
        for q, color, label in zip(quartiles, line_colors, line_labels):
            ax1.axvline(x=q, color=color, linestyle='--', linewidth=2, label=label)
        
        # Add title and labels
        ax1.set_title('Thickness Distribution')
        ax1.set_xlabel('Thickness')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Create metrics visualization (middle plot)
        ax2 = fig.add_subplot(gs[1])
        
        # Format metrics for display
        metric_names = []
        metric_values = []
        
        # Different metrics have different interpretations
        better_values = {
            'Radial Uniformity Index': 1.0,    # Higher is better (max = 1)
            'Gini Coefficient': 0.0,           # Lower is better (min = 0)
            'Thickness Range Ratio': 1.0       # Higher is better (max = 1)
        }
        
        # Sort metrics for display
        for name, value in metrics.items():
            metric_names.append(name)
            metric_values.append(value)
        
        # Create horizontal bar chart
        bars = ax2.barh(metric_names, metric_values, color='skyblue', edgecolor='black')
        
        # Add target lines
        for i, name in enumerate(metric_names):
            if name in better_values:
                target = better_values[name]
                ax2.axvline(x=target, ymin=(i/len(metric_names)), ymax=((i+1)/len(metric_names)), 
                          color='red', linestyle='--', linewidth=1)
        
        # Add title and labels
        ax2.set_title('Uniformity Metrics')
        ax2.set_xlabel('Value')
        
        # Add value annotations
        for i, v in enumerate(metric_values):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Adjust x-axis limits for better visualization
        ax2.set_xlim(0, max(max(metric_values) + 0.1, 1.0))
        
        # Add summary text (lower plot)
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')  # Hide axes
        
        # Create summary text
        summary = "Uniformity Analysis Summary:\n\n"
        
        # Add descriptions and interpretations
        descriptions = {
            'Radial Uniformity Index': 'Measures variation along radial lines. Higher values (closer to 1.0) indicate better uniformity.',
            'Gini Coefficient': 'Measures statistical dispersion. Lower values (closer to 0.0) indicate better uniformity.',
            'Thickness Range Ratio': 'Ratio of minimum to maximum thickness. Higher values (closer to 1.0) indicate better uniformity.'
        }
        
        for name, value in metrics.items():
            if name in descriptions:
                summary += f"• {name}: {value:.3f}\n  {descriptions[name]}\n\n"
        
        # Basic interpretation of overall uniformity
        radial_index = metrics.get('Radial Uniformity Index', 0)
        gini = metrics.get('Gini Coefficient', 1)
        range_ratio = metrics.get('Thickness Range Ratio', 0)
        
        if radial_index > 0.9 and gini < 0.1 and range_ratio > 0.9:
            overall = "Excellent uniformity across all metrics."
        elif radial_index > 0.7 and gini < 0.3 and range_ratio > 0.7:
            overall = "Good overall uniformity."
        elif radial_index > 0.5 and gini < 0.5 and range_ratio > 0.5:
            overall = "Moderate uniformity."
        else:
            overall = "Lower uniformity. Consider process optimization."
        
        summary += f"Overall Assessment: {overall}"
        
        # Add summary text to the plot
        ax3.text(0, 1, summary, va='top', ha='left', wrap=True)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
