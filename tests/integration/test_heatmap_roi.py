#!/usr/bin/env python3
"""
Test script to check heatmap ROI visualization for multiscale uniformity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
from esnf_mat_analyzer.visualization.visualization import Visualizer
from esnf_mat_analyzer.core.data_types import VisualizationConfig

def create_test_data_with_ruler():
    """Create test data that simulates a mat with a ruler on the side."""
    print("Creating test data with ruler simulation...")
    
    # Create a larger image to simulate ruler + mat
    height, width = 300, 400
    
    # Create base thickness map
    thickness_map = np.zeros((height, width))
    
    # Add a "ruler" region on the left (high contrast, regular pattern)
    ruler_width = 80
    thickness_map[:, :ruler_width] = 5.0  # High thickness for ruler
    
    # Add tick marks on ruler
    for i in range(10, height-10, 20):  # Every 20 pixels
        thickness_map[i-2:i+2, :ruler_width] = 8.0  # Bright tick marks
    
    # Add the "mat" region (right side with realistic nanofiber variation)
    mat_start = ruler_width + 20
    mat_region = thickness_map[:, mat_start:]
    
    # Create realistic mat thickness with some variation
    y, x = np.ogrid[:height, :mat_region.shape[1]]
    center_y, center_x = height // 2, mat_region.shape[1] // 2
    
    # Base thickness with some spatial variation
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    base_thickness = 2.0 + 0.5 * np.sin(distance * 0.1) + 0.3 * np.random.randn(height, mat_region.shape[1])
    base_thickness = np.clip(base_thickness, 0.5, 4.0)
    
    thickness_map[:, mat_start:] = base_thickness
    
    # Create ROI mask that excludes the ruler and focuses on the mat
    roi_mask = np.zeros((height, width), dtype=bool)
    
    # Circular ROI in the mat region only
    center_y_roi = height // 2
    center_x_roi = mat_start + (width - mat_start) // 2
    radius = min(height, width - mat_start) // 2 - 20
    
    y_roi, x_roi = np.ogrid[:height, :width]
    distance_roi = np.sqrt((y_roi - center_y_roi)**2 + (x_roi - center_x_roi)**2)
    roi_mask = distance_roi <= radius
    
    # Ensure ROI doesn't include ruler region
    roi_mask[:, :mat_start] = False
    
    return thickness_map, roi_mask

def test_heatmap_visualization():
    """Test the heatmap visualization to see what's being displayed."""
    print("Testing heatmap visualization...")
    
    # Create test data
    thickness_map, roi_mask = create_test_data_with_ruler()
    
    print(f"Data shapes: thickness_map {thickness_map.shape}, roi_mask {roi_mask.shape}")
    print(f"ROI coverage: {np.sum(roi_mask) / roi_mask.size * 100:.1f}%")
    print(f"Thickness range: {np.min(thickness_map):.3f} to {np.max(thickness_map):.3f}")
    
    roi_thickness = thickness_map[roi_mask]
    print(f"ROI thickness range: {np.min(roi_thickness):.3f} to {np.max(roi_thickness):.3f}")
    
    # Run multiscale analysis
    print("Running multiscale uniformity analysis...")
    analyzer = MultiScaleUniformityAnalyzer()
    results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
    
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")
    
    # Test visualization
    vis_config = VisualizationConfig()
    vis_config.auto_range_heatmap = True
    visualizer = Visualizer(vis_config)
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original thickness map
    im1 = axes[0, 0].imshow(thickness_map, cmap='viridis')
    axes[0, 0].set_title('Original Thickness Map\n(Including Ruler)')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # ROI mask
    axes[0, 1].imshow(roi_mask, cmap='gray')
    axes[0, 1].set_title('ROI Mask\n(Mat Region Only)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    
    # Masked thickness (what should be in heatmap)
    masked_thickness = np.ma.masked_array(thickness_map, mask=~roi_mask)
    im3 = axes[0, 2].imshow(masked_thickness, cmap='viridis')
    axes[0, 2].set_title('Masked Thickness\n(ROI Only)')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Test the actual visualizer heatmap
    heatmap_fig = visualizer.create_thickness_heatmap(thickness_map, roi_mask)
    
    # Extract the image data from the visualizer's figure to display in our grid
    # This is a bit hacky but helps us see what the visualizer actually produces
    axes[1, 0].text(0.5, 0.5, 'See separate\nVisualization\nWindow', 
                   ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
    axes[1, 0].set_title('Visualizer Output\n(See separate window)')
    axes[1, 0].axis('off')
    
    # Show distribution of values in ROI vs full image
    axes[1, 1].hist(thickness_map.flatten(), bins=50, alpha=0.7, label='Full Image', color='blue')
    axes[1, 1].hist(roi_thickness, bins=50, alpha=0.7, label='ROI Only', color='red')
    axes[1, 1].set_title('Thickness Distribution')
    axes[1, 1].set_xlabel('Thickness')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Show multiscale results
    if results:
        scale_names = list(results.keys())
        scale_values = list(results.values())
        
        axes[1, 2].bar(range(len(scale_names)), scale_values, color='skyblue', edgecolor='black')
        axes[1, 2].set_title('Multiscale Uniformity Results')
        axes[1, 2].set_xlabel('Scale Level')
        axes[1, 2].set_ylabel('Uniformity Score')
        axes[1, 2].set_xticks(range(len(scale_names)))
        axes[1, 2].set_xticklabels([f'S{i}' for i in range(len(scale_names))], rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(scale_values):
            axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('heatmap_roi_test_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved to: heatmap_roi_test_comparison.png")
    
    # Show the visualizer's heatmap in a separate window
    heatmap_fig.suptitle('Visualizer Heatmap Output\n(Should show only ROI region)')
    plt.show()

if __name__ == "__main__":
    test_heatmap_visualization()
    print("Test completed!")
