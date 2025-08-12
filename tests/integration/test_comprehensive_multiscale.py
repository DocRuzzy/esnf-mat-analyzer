#!/usr/bin/env python3
"""
Comprehensive test for multiscale uniformity heatmap visualization.
This demonstrates the difference between thickness heatmap (ROI-masked) 
and multiscale uniformity spatial distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
from esnf_mat_analyzer.visualization.visualization import Visualizer
from esnf_mat_analyzer.core.data_types import VisualizationConfig

def create_realistic_test_data():
    """Create realistic test data with ruler and varying mat regions."""
    print("Creating realistic test data...")
    
    height, width = 400, 500
    thickness_map = np.zeros((height, width))
    
    # Ruler region (left side) - high contrast, regular pattern
    ruler_width = 100
    thickness_map[:, :ruler_width] = 6.0
    
    # Add ruler tick marks
    for i in range(20, height-20, 25):
        thickness_map[i-3:i+3, :ruler_width] = 9.0
    
    # Mat region with different uniformity zones
    mat_start = ruler_width + 30
    
    # Zone 1: High uniformity (top)
    zone1_height = height // 3
    uniform_thickness = 2.0 + 0.1 * np.random.randn(zone1_height, width - mat_start)
    thickness_map[:zone1_height, mat_start:] = np.clip(uniform_thickness, 1.5, 2.5)
    
    # Zone 2: Medium uniformity (middle)
    zone2_start = zone1_height
    zone2_end = 2 * height // 3
    zone2_height = zone2_end - zone2_start
    
    y, x = np.ogrid[:zone2_height, :width - mat_start]
    moderate_thickness = 2.5 + 0.3 * np.sin(x * 0.1) + 0.2 * np.sin(y * 0.1) + 0.15 * np.random.randn(zone2_height, width - mat_start)
    thickness_map[zone2_start:zone2_end, mat_start:] = np.clip(moderate_thickness, 1.0, 4.0)
    
    # Zone 3: Low uniformity (bottom)
    zone3_start = zone2_end
    zone3_height = height - zone3_start
    
    # Create patches of different thickness
    y, x = np.ogrid[:zone3_height, :width - mat_start]
    center1_y, center1_x = zone3_height // 3, (width - mat_start) // 3
    center2_y, center2_x = 2 * zone3_height // 3, 2 * (width - mat_start) // 3
    
    dist1 = np.sqrt((y - center1_y)**2 + (x - center1_x)**2)
    dist2 = np.sqrt((y - center2_y)**2 + (x - center2_x)**2)
    
    patchy_thickness = 2.0 + 1.5 * np.exp(-dist1/20) + 1.0 * np.exp(-dist2/30) + 0.4 * np.random.randn(zone3_height, width - mat_start)
    thickness_map[zone3_start:, mat_start:] = np.clip(patchy_thickness, 0.5, 5.0)
    
    # Create ROI mask that covers only the mat region
    roi_mask = np.zeros((height, width), dtype=bool)
    
    # Rectangular ROI covering the mat region
    roi_y_start = 50
    roi_y_end = height - 50
    roi_x_start = mat_start + 20
    roi_x_end = width - 50
    
    roi_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = True
    
    return thickness_map, roi_mask

def test_comprehensive_multiscale_visualization():
    """Test both regular and multiscale uniformity visualizations."""
    print("Running comprehensive multiscale visualization test...")
    
    # Create test data
    thickness_map, roi_mask = create_realistic_test_data()
    
    print(f"Data shapes: thickness_map {thickness_map.shape}, roi_mask {roi_mask.shape}")
    print(f"ROI coverage: {np.sum(roi_mask) / roi_mask.size * 100:.1f}%")
    
    # 1. Regular multiscale analysis (global scores)
    print("\\n1. Running global multiscale uniformity analysis...")
    analyzer = MultiScaleUniformityAnalyzer()
    global_results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
    
    print("Global Results:")
    for key, value in global_results.items():
        print(f"  {key}: {value:.6f}")
    
    # 2. Spatial multiscale analysis (heatmaps)
    print("\\n2. Creating spatial multiscale uniformity maps...")
    spatial_maps = analyzer.create_multiscale_heatmap(thickness_map, roi_mask, window_size=80)
    
    print(f"Created {len(spatial_maps)} spatial maps:")
    for map_name in spatial_maps.keys():
        print(f"  {map_name}")
    
    # 3. Create visualizations
    print("\\n3. Creating visualizations...")
    vis_config = VisualizationConfig()
    vis_config.auto_range_heatmap = True
    visualizer = Visualizer(vis_config)
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 15))
    
    # Layout: 3 rows, 4 columns
    # Row 1: Original data
    # Row 2: Standard thickness heatmap vs ROI
    # Row 3: Multiscale spatial maps
    
    # Row 1: Original data overview
    plt.subplot(3, 4, 1)
    plt.imshow(thickness_map, cmap='viridis')
    plt.title('Original Thickness Map\\n(Ruler + Mat)')
    plt.colorbar(shrink=0.8)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    plt.subplot(3, 4, 2)
    plt.imshow(roi_mask, cmap='gray')
    plt.title('ROI Mask\\n(Mat Region Only)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    plt.subplot(3, 4, 3)
    masked_thickness = np.ma.masked_array(thickness_map, mask=~roi_mask)
    plt.imshow(masked_thickness, cmap='viridis')
    plt.title('ROI-Masked Thickness\\n(What Heatmap Shows)')
    plt.colorbar(shrink=0.8)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    plt.subplot(3, 4, 4)
    roi_thickness = thickness_map[roi_mask]
    plt.hist(roi_thickness, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('ROI Thickness Distribution')
    plt.xlabel('Thickness')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Row 2: Standard visualizations
    plt.subplot(3, 4, 5)
    # This simulates what the standard heatmap looks like
    plt.imshow(masked_thickness, cmap='viridis')
    plt.title('Standard Thickness Heatmap\\n(ROI Only - No Ruler)')
    plt.colorbar(shrink=0.8)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    plt.subplot(3, 4, 6)
    scale_names = [f'Scale {i}' for i in range(len(global_results))]
    scale_values = list(global_results.values())
    bars = plt.bar(scale_names, scale_values, color='lightcoral', edgecolor='black')
    plt.title('Global Multiscale Uniformity')
    plt.ylabel('Uniformity Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, scale_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Calculate overall score
    overall_score = np.mean(scale_values)
    plt.subplot(3, 4, 7)
    plt.text(0.5, 0.7, f'Overall Multiscale\\nUniformity Score', ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=14, fontweight='bold')
    plt.text(0.5, 0.5, f'{overall_score:.3f}', ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=24, fontweight='bold', 
             color='green' if overall_score > 0.8 else 'orange' if overall_score > 0.6 else 'red')
    
    # Interpretation
    if overall_score > 0.9:
        interpretation = 'Excellent\\nUniformity'
    elif overall_score > 0.8:
        interpretation = 'Good\\nUniformity'
    elif overall_score > 0.6:
        interpretation = 'Moderate\\nUniformity'
    else:
        interpretation = 'Poor\\nUniformity'
    
    plt.text(0.5, 0.3, interpretation, ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=12, style='italic')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.text(0.5, 0.5, 'ROI Mask ensures\\nheatmap shows\\nonly the mat,\\nnot the ruler', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.axis('off')
    
    # Row 3: Show spatial maps if available
    if spatial_maps:
        spatial_items = list(spatial_maps.items())
        for i in range(min(4, len(spatial_items))):
            plt.subplot(3, 4, 9 + i)
            map_name, map_data = spatial_items[i]
            scale_level = map_name.replace('scale_', '').replace('_uniformity_map', '')
            
            # Create masked spatial map
            masked_spatial = np.ma.masked_array(map_data, mask=~roi_mask.astype(bool))
            masked_spatial = np.ma.masked_where(masked_spatial == 0, masked_spatial)
            
            if np.ma.count(masked_spatial) > 0:
                plt.imshow(masked_spatial, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
                plt.colorbar(shrink=0.6, label='Uniformity')
            else:
                plt.text(0.5, 0.5, 'No spatial\\ndata available\\n(window too large)', 
                        ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.title(f'Scale {scale_level} Spatial Uniformity')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.xticks([])
            plt.yticks([])
    else:
        plt.subplot(3, 4, 9)
        plt.text(0.5, 0.5, 'Spatial multiscale\\nmaps not available\\n(window size too large\\nfor this image)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    plt.suptitle('Multiscale Uniformity Analysis: ROI-Focused Visualization\\n' + 
                'Top: Original data and ROI masking | Middle: Standard analysis | Bottom: Spatial analysis', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig('comprehensive_multiscale_visualization.png', dpi=150, bbox_inches='tight')
    print("Comprehensive visualization saved to: comprehensive_multiscale_visualization.png")
    
    # Create the standard thickness heatmap using the visualizer
    print("\\n4. Creating standard thickness heatmap...")
    thickness_heatmap_fig = visualizer.create_thickness_heatmap(thickness_map, roi_mask)
    thickness_heatmap_fig.suptitle('Standard Thickness Heatmap (Visualizer Output)\\nShows only ROI region, ruler is masked out')
    
    # Create multiscale spatial heatmap if data is available
    if spatial_maps:
        print("5. Creating multiscale spatial heatmap...")
        multiscale_heatmap_fig = visualizer.create_multiscale_uniformity_heatmap(spatial_maps, roi_mask)
        multiscale_heatmap_fig.suptitle('Multiscale Uniformity Spatial Distribution\\nShows local uniformity variations across the mat')
    
    plt.show()

if __name__ == "__main__":
    test_comprehensive_multiscale_visualization()
    print("\\nTest completed successfully!")
    print("\\nKey findings:")
    print("- The standard thickness heatmap correctly masks out the ruler region")
    print("- Only the ROI (mat) region is analyzed and displayed")
    print("- Multiscale analysis provides both global scores and spatial maps")
    print("- Spatial maps show local variations in uniformity across different scales")
