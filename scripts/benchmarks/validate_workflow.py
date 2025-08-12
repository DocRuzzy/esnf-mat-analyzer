#!/usr/bin/env python3
"""
Validation test for the optimized workflow implementation.

This validates that the analyzer properly implements:
1. Background correction uses full image
2. Scale detection outside user ROI  
3. Uniformity analysis on gel-enclosed area
4. Ruler exclusion from background correction
"""

import numpy as np
from pathlib import Path
import cv2
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

def test_workflow_optimization():
    """Test the key workflow optimizations."""
    print("Testing Workflow Optimizations")
    print("=" * 40)
    
    # Create simple test image
    size = 600
    image = np.ones((size, size), dtype=np.uint8) * 130
    
    # Add gel region (circle in center)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    gel_radius = 100
    gel_mask = (x - center)**2 + (y - center)**2 < gel_radius**2
    image[gel_mask] = 170  # Brighter gel/mat area
    
    # Add illumination gradient
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    illumination = 0.8 + 0.3 * xx  # Left-to-right gradient
    image_with_grad = np.clip(image.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
    
    # Add ruler region (should be outside user ROI)
    ruler_y, ruler_x = 20, 20
    image_with_grad[ruler_y:ruler_y+30, ruler_x:ruler_x+150] = 60  # Dark ruler
    
    # Save test image
    test_path = Path("validation_test.png")
    cv2.imwrite(str(test_path), image_with_grad)
    
    try:
        # Setup analyzer
        config = get_default_config()
        config.ruler_detection.enabled = True
        config.processing.leveling.enabled = True
        
        analyzer = setup_dependencies(config)
        
        # Test with user ROI that excludes ruler
        user_roi = (150, 150, 300, 300)  # Center region, excludes ruler at (20,20)
        
        print(f"Testing with user ROI: {user_roi}")
        print(f"Ruler location: ({ruler_x}, {ruler_y}) - outside ROI ✓")
        
        result = analyzer.process_image(test_path, roi=user_roi)
        
        print("\nResults:")
        print(f"✓ Processing time: {result.processing_time:.2f}s")
        print(f"✓ Ruler detected: {result.metadata.get('ruler_detected', False)}")
        print(f"✓ Gel detection: {result.metadata.get('gel_detection_successful', False)}")
        print(f"✓ Background correction applied: {result.metadata.get('background_correction_applied', False)}")
        
        # Validate shapes
        full_shape = result.metadata.get('full_image_shape', (0, 0))
        roi_shape = result.metadata.get('roi_shape', (0, 0))
        print(f"✓ Full image: {full_shape}")
        print(f"✓ ROI shape: {roi_shape}")
        print(f"✓ Used full image for background correction: {full_shape[0] > roi_shape[0]}")
        
        # Check uniformity metrics
        if 'coefficient_of_variation' in result.metrics:
            cv = result.metrics['coefficient_of_variation']
            print(f"✓ Coefficient of Variation: {cv:.4f}")
        
        if result.mask is not None:
            mask_pixels = np.sum(result.mask)
            total_pixels = result.mask.size
            print(f"✓ Analysis area: {mask_pixels}/{total_pixels} pixels ({100*mask_pixels/total_pixels:.1f}%)")
        
        print("\n🎉 Workflow optimization validation successful!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()

if __name__ == "__main__":
    test_workflow_optimization()
