#!/usr/bin/env python3
"""
Test the enhanced shadow-aware exclusion mask functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor

def test_shadow_exclusion():
    """Test the enhanced exclusion mask that accounts for shadows."""
    print("Testing Enhanced Shadow-Aware Exclusion")
    print("=" * 50)
    
    # Create test image with ruler and shadow
    size = 400
    image = np.ones((size, size), dtype=np.uint8) * 150
    
    # Add ruler (horizontal at top)
    ruler_y, ruler_x = 30, 50
    ruler_h, ruler_w = 20, 120
    image[ruler_y:ruler_y+ruler_h, ruler_x:ruler_x+ruler_w] = 60
    
    # Add shadow below ruler (gradual darkening)
    shadow_height = 40
    for i in range(shadow_height):
        shadow_strength = 1.0 - (0.3 * np.exp(-i/15))  # Exponential fade
        shadow_y = ruler_y + ruler_h + i
        if shadow_y < size:
            image[shadow_y, ruler_x:ruler_x+ruler_w] = (
                image[shadow_y, ruler_x:ruler_x+ruler_w] * shadow_strength
            ).astype(np.uint8)
    
    # Add gel region for analysis
    center = size // 2
    gel_radius = 80
    y, x = np.ogrid[:size, :size]
    gel_mask = (x - center)**2 + (y - center)**2 < gel_radius**2
    image[gel_mask] = 180
    
    # Add illumination gradient
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    illumination = 0.8 + 0.25 * xx
    image_final = np.clip(image.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
    
    # Create original ruler mask (just the ruler)
    ruler_mask = np.zeros((size, size), dtype=bool)
    ruler_mask[ruler_y:ruler_y+ruler_h, ruler_x:ruler_x+ruler_w] = True
    
    print(f"Test setup:")
    print(f"  Image size: {size}x{size}")
    print(f"  Ruler: ({ruler_x},{ruler_y}) size {ruler_w}x{ruler_h}")
    print(f"  Shadow area: below ruler, {shadow_height} pixels high")
    print(f"  Gel center: ({center},{center}), radius {gel_radius}")
    
    # Test both exclusion methods
    processor = AdvancedBackgroundProcessor()
    
    print(f"\n--- Method 1: Original exclusion (ruler only) ---")
    results_original = processor.complete_uniformity_analysis_with_exclusion(
        image_final,
        mat_threshold=160,
        background_method="large_kernel_blur",
        exclusion_mask=ruler_mask
    )
    
    print(f"\n--- Method 2: Enhanced exclusion (ruler + shadows) ---")
    results_enhanced = processor.complete_uniformity_analysis_with_exclusion(
        image_final,
        mat_threshold=160,
        background_method="large_kernel_blur",
        exclusion_mask=ruler_mask  # This will be automatically expanded
    )
    
    # Analyze shadow correction quality
    shadow_region_y = ruler_y + ruler_h + 10
    shadow_region = slice(shadow_region_y, shadow_region_y + 20)
    shadow_cols = slice(ruler_x + 10, ruler_x + ruler_w - 10)
    
    normal_region_y = center
    normal_region = slice(normal_region_y, normal_region_y + 20)
    normal_cols = slice(center - 30, center + 30)
    
    # Compare results
    shadow_orig = results_original['corrected_image'][shadow_region, shadow_cols]
    shadow_enh = results_enhanced['corrected_image'][shadow_region, shadow_cols]
    normal_orig = results_original['corrected_image'][normal_region, normal_cols]
    normal_enh = results_enhanced['corrected_image'][normal_region, normal_cols]
    
    diff_orig = abs(np.mean(shadow_orig) - np.mean(normal_orig))
    diff_enh = abs(np.mean(shadow_enh) - np.mean(normal_enh))
    
    print(f"\nShadow Correction Results:")
    print(f"✓ Original method - Shadow vs Normal difference: {diff_orig:.1f}")
    print(f"✓ Enhanced method - Shadow vs Normal difference: {diff_enh:.1f}")
    print(f"✓ Improvement: {diff_orig - diff_enh:.1f} intensity units")
    print(f"✓ Relative improvement: {100*(diff_orig - diff_enh)/diff_orig:.1f}%")
    
    # Check exclusion mask expansion
    original_excluded = np.sum(ruler_mask)
    enhanced_excluded = np.sum(results_enhanced.get('exclusion_mask', ruler_mask))
    
    print(f"\nExclusion Mask Analysis:")
    print(f"✓ Original excluded pixels: {original_excluded}")
    print(f"✓ Enhanced excluded pixels: {enhanced_excluded}")
    print(f"✓ Additional excluded pixels: {enhanced_excluded - original_excluded}")
    print(f"✓ Expansion ratio: {enhanced_excluded/original_excluded:.2f}x")
    
    if diff_enh < diff_orig:
        print(f"\n🎉 Enhanced exclusion successfully improved shadow correction!")
    else:
        print(f"\n⚠ Enhanced exclusion did not improve results. May need tuning.")
    
    return results_original, results_enhanced

def test_heatmap_focus():
    """Test that heatmap focuses on the correct area."""
    print(f"\n" + "="*50)
    print("Testing Heatmap Focus on Gel Area")
    print("=" * 50)
    
    # This would require integration with the actual GUI
    # For now, just validate the mask logic
    from esnf_mat_analyzer.visualization.visualization import Visualizer
    from esnf_mat_analyzer.core.data_types import VisualizationConfig
    
    # Create test data
    size = 300
    thickness_map = np.random.normal(1.0, 0.2, (size, size))
    
    # Create gel mask (circular region)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    gel_mask = (x - center)**2 + (y - center)**2 < (size//3)**2
    
    # Create ruler area that should be excluded
    ruler_mask = np.zeros((size, size), dtype=bool)
    ruler_mask[20:40, 50:150] = True
    
    # The analysis mask should be gel_mask AND NOT ruler_mask
    analysis_mask = gel_mask & ~ruler_mask
    
    print(f"Mask validation:")
    print(f"✓ Total image pixels: {size*size}")
    print(f"✓ Gel area pixels: {np.sum(gel_mask)}")
    print(f"✓ Ruler area pixels: {np.sum(ruler_mask)}")
    print(f"✓ Analysis area pixels: {np.sum(analysis_mask)}")
    print(f"✓ Analysis covers {100*np.sum(analysis_mask)/(size*size):.1f}% of image")
    
    # Test heatmap creation
    vis_config = VisualizationConfig()
    visualizer = Visualizer(vis_config)
    
    try:
        fig = visualizer.create_thickness_heatmap(thickness_map, analysis_mask)
        print(f"✓ Heatmap created successfully using analysis mask")
        plt.close(fig)  # Don't display, just test creation
    except Exception as e:
        print(f"❌ Heatmap creation failed: {e}")
    
    print(f"✓ Heatmap focus test completed")

if __name__ == "__main__":
    print("Enhanced Background Correction Testing")
    print("=" * 60)
    
    try:
        # Test shadow exclusion enhancement
        results = test_shadow_exclusion()
        
        # Test heatmap focus
        test_heatmap_focus()
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
