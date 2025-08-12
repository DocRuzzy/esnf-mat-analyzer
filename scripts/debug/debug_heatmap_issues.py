#!/usr/bin/env python3
"""
Debug script to investigate the heatmap and background correction issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

def debug_analysis_results():
    """Debug what masks and regions are being used in the analysis."""
    print("Debugging Analysis Results")
    print("=" * 40)
    
    # Create test image that mimics the real issue
    size = 600
    image = np.ones((size, size), dtype=np.uint8) * 140  # Base background
    
    # Add gel region (large circular area)
    center = size // 2
    gel_radius = 120
    y, x = np.ogrid[:size, :size]
    gel_mask = (x - center)**2 + (y - center)**2 < gel_radius**2
    image[gel_mask] = 180  # Gel/mat area
    
    # Add ruler at top
    ruler_y, ruler_x = 30, 100
    ruler_h, ruler_w = 25, 150
    image[ruler_y:ruler_y+ruler_h, ruler_x:ruler_x+ruler_w] = 60  # Dark ruler
    
    # Add scale markings
    for i in range(0, ruler_w, 15):
        if ruler_x + i + 2 < size:
            image[ruler_y:ruler_y+5, ruler_x+i:ruler_x+i+2] = 40
    
    # Add shaded area below ruler (this is the problem area)
    shadow_y = ruler_y + ruler_h
    shadow_h = 60
    shadow_gradient = np.linspace(0.9, 1.0, shadow_h)[:, np.newaxis]
    for i in range(shadow_h):
        if shadow_y + i < size:
            shadow_strength = shadow_gradient[i, 0]
            image[shadow_y + i, ruler_x:ruler_x+ruler_w] = (
                image[shadow_y + i, ruler_x:ruler_x+ruler_w] * shadow_strength
            ).astype(np.uint8)
    
    # Add illumination gradient
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    illumination = 0.85 + 0.2 * xx  # Left-to-right gradient
    image_with_grad = np.clip(image.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
    
    # Save test image
    test_path = Path("debug_test.png")
    cv2.imwrite(str(test_path), image_with_grad)
    
    try:
        # Setup analyzer
        config = get_default_config()
        config.ruler_detection.enabled = True
        config.processing.leveling.enabled = True
        
        analyzer = setup_dependencies(config)
        
        # Test with user ROI that includes gel but excludes ruler
        user_roi = (center-150, center-150, 300, 300)  # Center around gel
        
        print(f"User ROI: {user_roi}")
        print(f"Gel center: ({center}, {center})")
        print(f"Ruler location: ({ruler_x}, {ruler_y})")
        print(f"Shadow area: ({ruler_x}, {shadow_y}) to ({ruler_x+ruler_w}, {shadow_y+shadow_h})")
        
        # Run analysis
        result = analyzer.process_image(test_path, roi=user_roi)
        
        print(f"\nAnalysis Results:")
        print(f"✓ Processing time: {result.processing_time:.2f}s")
        print(f"✓ Ruler detected: {result.metadata.get('ruler_detected', False)}")
        print(f"✓ Gel detection: {result.metadata.get('gel_detection_successful', False)}")
        
        # Check what mask is being used
        if result.mask is not None:
            mask_pixels = np.sum(result.mask)
            total_roi_pixels = result.mask.size
            print(f"✓ Mask area: {mask_pixels}/{total_roi_pixels} pixels ({100*mask_pixels/total_roi_pixels:.1f}%)")
            
            # Check if mask covers expected gel area
            roi_x, roi_y, roi_w, roi_h = user_roi
            expected_gel_in_roi = gel_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            expected_gel_pixels = np.sum(expected_gel_in_roi)
            overlap = np.sum(result.mask & expected_gel_in_roi)
            
            print(f"✓ Expected gel pixels in ROI: {expected_gel_pixels}")
            print(f"✓ Actual mask overlap with gel: {overlap} pixels")
            print(f"✓ Mask accuracy: {100*overlap/expected_gel_pixels:.1f}%")
        
        # Check thickness map range
        if result.thickness_map is not None:
            thickness_min = np.min(result.thickness_map)
            thickness_max = np.max(result.thickness_map)
            thickness_mean = np.mean(result.thickness_map)
            print(f"✓ Thickness range: {thickness_min:.1f} - {thickness_max:.1f} (mean: {thickness_mean:.1f})")
        
        # Check if shadow area is properly corrected
        roi_x, roi_y, roi_w, roi_h = user_roi
        roi_corrected = result.thickness_map  # This should be the corrected image data
        
        # Look at the area where shadow should be
        shadow_in_roi_y = max(0, shadow_y - roi_y)
        shadow_in_roi_x = max(0, ruler_x - roi_x)
        
        if (shadow_in_roi_y < roi_h and shadow_in_roi_x < roi_w and 
            shadow_in_roi_y + 20 < roi_h and shadow_in_roi_x + 50 < roi_w):
            
            shadow_region = roi_corrected[shadow_in_roi_y:shadow_in_roi_y+20, 
                                        shadow_in_roi_x:shadow_in_roi_x+50]
            normal_region = roi_corrected[roi_h//2:roi_h//2+20, roi_w//2:roi_w//2+50]
            
            shadow_mean = np.mean(shadow_region)
            normal_mean = np.mean(normal_region)
            difference = abs(shadow_mean - normal_mean)
            
            print(f"✓ Shadow region mean: {shadow_mean:.1f}")
            print(f"✓ Normal region mean: {normal_mean:.1f}")
            print(f"✓ Difference: {difference:.1f}")
            print(f"✓ Shadow correction quality: {'Good' if difference < 10 else 'Poor'}")
        
        # Test both background correction methods
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        
        print(f"\n--- Testing Background Correction Methods ---")
        
        # Create ruler mask for exclusion testing
        ruler_mask = np.zeros((size, size), dtype=bool)
        ruler_mask[ruler_y:ruler_y+ruler_h, ruler_x:ruler_x+ruler_w] = True
        
        # Test polynomial method
        results_poly = processor.complete_uniformity_analysis_with_exclusion(
            image_with_grad,
            mat_threshold=160,
            background_method="polynomial",
            exclusion_mask=ruler_mask
        )
        
        # Test kernel blur method  
        results_blur = processor.complete_uniformity_analysis_with_exclusion(
            image_with_grad,
            mat_threshold=160,
            background_method="large_kernel_blur",
            exclusion_mask=ruler_mask
        )
        
        # Compare shadow correction in both methods
        shadow_poly = results_poly['corrected_image'][shadow_y:shadow_y+20, ruler_x:ruler_x+50]
        shadow_blur = results_blur['corrected_image'][shadow_y:shadow_y+20, ruler_x:ruler_x+50]
        normal_poly = results_poly['corrected_image'][center:center+20, center:center+50]
        normal_blur = results_blur['corrected_image'][center:center+20, center:center+50]
        
        poly_shadow_diff = abs(np.mean(shadow_poly) - np.mean(normal_poly))
        blur_shadow_diff = abs(np.mean(shadow_blur) - np.mean(normal_blur))
        
        print(f"✓ Polynomial shadow correction: {poly_shadow_diff:.1f} intensity difference")
        print(f"✓ Kernel blur shadow correction: {blur_shadow_diff:.1f} intensity difference")
        print(f"✓ Better method: {'Polynomial' if poly_shadow_diff < blur_shadow_diff else 'Kernel Blur'}")
        
        return result, results_poly, results_blur
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()

if __name__ == "__main__":
    debug_analysis_results()
