
"""
Visualization components for nanofiber thickness analysis.

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)

This module implements various visualization techniques for thickness maps
and uniformity metrics, including heatmaps, radial profiles, and metric plots.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import Tuple, Dict, Optional, Any, List
import logging

from esnf_mat_analyzer.core.interfaces import VisualizerInterface
from esnf_mat_analyzer.core.data_types import VisualizationConfig

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
        
        FFT enhancement is now OPTIONAL and controlled by config.fft_enhancement_enabled.
        By default, FFT enhancement is disabled to preserve real thickness features.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            saturation_mask: Optional mask of saturated pixels
            
        Returns:
            Matplotlib Figure object
        """
        # Apply FFT enhancement only if explicitly enabled
        if self.config.fft_enhancement_enabled and self.config.fft_blend_ratio > 0:
            self.logger.debug("Creating thickness heatmap with FFT enhancement")
            enhanced_map = self._enhance_with_fft(thickness_map, mask)
            title_suffix = "(FFT-Enhanced)"
        else:
            self.logger.debug("Creating thickness heatmap WITHOUT FFT enhancement (preserving all features)")
            enhanced_map = thickness_map.copy()
            title_suffix = ""

        # Mild display smoothing to improve perceived gradient without flattening data
        try:
            smoothed_map = cv2.GaussianBlur(enhanced_map.astype(np.float32), (7, 7), sigmaX=1.2, sigmaY=1.2)
        except Exception:
            smoothed_map = enhanced_map
        
        # Create masked thickness map (only show values within the mask)
        masked_thickness = np.ma.masked_array(smoothed_map, mask=~(mask.astype(bool)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Use specified colormap
        cmap = plt.get_cmap(self.config.colormap)
        
        # Create heatmap
        # Calculate reasonable color limits based on the actual data
        if self.config.auto_range_heatmap:
            valid_data = masked_thickness.compressed()  # Get non-masked values
            if len(valid_data) > 0:
                # Use configurable percentiles to avoid extreme outliers affecting the color scale
                # Widen the percentile window slightly for smoother color gradients
                min_percentile, max_percentile = (max(0, self.config.heatmap_percentile_range[0]-1),
                                                  min(100, self.config.heatmap_percentile_range[1]+1))
                vmin = np.percentile(valid_data, min_percentile)
                vmax = np.percentile(valid_data, max_percentile)
                
                # Log data statistics
                self.logger.info(f"FFT-enhanced thickness data - Min: {np.min(valid_data):.3f}, Max: {np.max(valid_data):.3f}, Mean: {np.mean(valid_data):.3f}")
                self.logger.info(f"Color range (percentiles {min_percentile}-{max_percentile}): [{vmin:.3f}, {vmax:.3f}]")
            else:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None
        
        # Use bilinear interpolation for smoother visual gradients
        im = ax.imshow(masked_thickness, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        
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
        ax.set_title(f'Nanofiber Thickness Map {title_suffix}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        return fig
    
    def _enhance_with_fft(self, thickness_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Enhance thickness map using FFT to reveal high-frequency variations.
        
        WARNING: This method can remove real thickness features along with background.
        Use with caution and verify results against known patterns.
        
        This method:
        1. Applies FFT to decompose the thickness map
        2. Enhances high-frequency components (variations)
        3. Reduces low-frequency components (smooth gradients)
        4. Reconstructs the enhanced map
        
        Parameters are controlled by config:
        - fft_blend_ratio: How much to blend FFT result with original (0=none, 1=full)
        - fft_sigma_divisor: Controls high-pass filter width (larger=gentler)
        - fft_amplification: Amplification of high frequencies
        
        Args:
            thickness_map: Original thickness map
            mask: Region of interest mask
            
        Returns:
            FFT-enhanced thickness map (blended with original per config)
        """
        # Create a copy to work with
        enhanced = thickness_map.copy().astype(np.float32)
        
        # Get config parameters with safe defaults
        blend_ratio = getattr(self.config, 'fft_blend_ratio', 0.85)
        sigma_divisor = getattr(self.config, 'fft_sigma_divisor', 12.0)
        amplification = getattr(self.config, 'fft_amplification', 2.0)
        
        # Only process within the masked region
        if mask is not None and np.any(mask):
            # Get valid region
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0:
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                
                # Extract ROI
                roi = enhanced[y_min:y_max+1, x_min:x_max+1].copy()
                roi_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
                
                # Apply FFT
                fft_result = np.fft.fft2(roi)
                fft_shifted = np.fft.fftshift(fft_result)
                
                # Create high-pass filter with configurable sigma
                h, w = roi.shape
                cy, cx = h // 2, w // 2
                
                # Gaussian high-pass filter - larger sigma_divisor = gentler filtering
                y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
                sigma = min(h, w) / sigma_divisor
                gaussian_hp = 1 - np.exp(-(x**2 + y**2) / (2 * sigma**2))
                
                # Apply filter with configurable amplification
                fft_filtered = fft_shifted * gaussian_hp * amplification
                
                # Inverse FFT
                fft_ishift = np.fft.ifftshift(fft_filtered)
                roi_enhanced = np.fft.ifft2(fft_ishift).real
                
                # Normalize to original range
                roi_min, roi_max = roi[roi_mask > 0].min(), roi[roi_mask > 0].max()
                roi_enhanced_norm = (roi_enhanced - roi_enhanced.min()) / (roi_enhanced.max() - roi_enhanced.min() + 1e-8)
                roi_enhanced_norm = roi_enhanced_norm * (roi_max - roi_min) + roi_min
                
                # Blend with original using configurable ratio
                roi_blended = blend_ratio * roi_enhanced_norm + (1 - blend_ratio) * roi
                
                # Replace in full map
                enhanced[y_min:y_max+1, x_min:x_max+1] = roi_blended
                
                self.logger.info(f"Applied FFT enhancement (blend={blend_ratio:.2f}, sigma_div={sigma_divisor:.1f}, amp={amplification:.1f})")
        
        return enhanced
    
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

    def create_multiscale_uniformity_heatmap(self, multiscale_maps: Dict[str, np.ndarray], 
                                           roi_mask: np.ndarray) -> Figure:
        """
        Create a visualization of multiscale uniformity spatial maps.
        
        Args:
            multiscale_maps: Dictionary of spatial uniformity maps for each scale
            roi_mask: Boolean mask indicating the region of interest
            
        Returns:
            Matplotlib Figure object
        """
        self.logger.debug("Creating multiscale uniformity heatmap")
        
        # Filter maps that are actually spatial (have '_map' in name)
        spatial_maps = {k: v for k, v in multiscale_maps.items() if '_map' in k}
        
        if not spatial_maps:
            # No spatial maps available, create a placeholder
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No spatial multiscale maps available.\nRun analysis with spatial mapping enabled.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Multiscale Uniformity Spatial Maps')
            ax.axis('off')
            return fig
        
        # Determine grid layout
        n_maps = len(spatial_maps)
        cols = min(3, n_maps)  # Max 3 columns
        rows = (n_maps + cols - 1) // cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(self.config.figure_size[0] * cols / 2, 
                                                     self.config.figure_size[1] * rows / 2))
        
        # Handle single subplot case
        if n_maps == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each scale map
        for idx, (map_name, map_data) in enumerate(spatial_maps.items()):
            ax = axes[idx]
            
            # Extract scale level from map name
            scale_level = map_name.replace('scale_', '').replace('_uniformity_map', '')
            
            # Create masked array for better visualization
            masked_data = np.ma.masked_array(map_data, mask=~roi_mask.astype(bool))
            
            # Only show non-zero values (areas that were actually analyzed)
            masked_data = np.ma.masked_where(masked_data == 0, masked_data)
            
            # Create heatmap
            if np.ma.count(masked_data) > 0:  # If there's data to show
                im = ax.imshow(masked_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
                plt.colorbar(im, ax=ax, shrink=0.8, label='Uniformity Score')
            else:
                # No data to show, create placeholder
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.imshow(np.zeros_like(roi_mask), cmap='gray', alpha=0.3)
            
            ax.set_title(f'Scale {scale_level} Uniformity')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(n_maps, len(axes)):
            axes[idx].axis('off')
        
        # Add overall title
        fig.suptitle('Multiscale Uniformity Spatial Distribution\n(Green = High Uniformity, Red = Low Uniformity)', 
                    fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
