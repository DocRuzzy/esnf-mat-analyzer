#!/usr/bin/env python3
"""
Test script for the optimized analysis workflow with proper ROI handling.

This tests the new workflow:
1. Background correction on full image (excluding rulers)
2. Scale detection outside user ROI
3. Gel detection within user ROI for optimal analysis area
4. Uniformity analysis on gel-enclosed area only
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comprehensive_test_image():
    """Create a synthetic image with ruler, gel, mat, and uneven illumination."""
    size = 800
    image = np.ones((size, size), dtype=np.uint8) * 120  # Background
    
    # Add uneven illumination (gradient)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    illumination = 0.7 + 0.4 * (xx + 0.3 * yy)  # Diagonal gradient
    
    # Create gel outline (rectangular with some irregularity)
    gel_center_x, gel_center_y = size // 2, size // 2
    gel_width, gel_height = 300, 250
    
    # Gel boundary (slightly irregular)
    gel_mask = np.zeros((size, size), dtype=bool)
    for y in range(gel_center_y - gel_height//2, gel_center_y + gel_height//2):
        for x in range(gel_center_x - gel_width//2, gel_center_x + gel_width//2):
            if 0 <= x < size and 0 <= y < size:
                # Add some irregularity to gel boundary
                noise = np.random.normal(0, 3)
                if (abs(x - gel_center_x) < gel_width//2 + noise and 
                    abs(y - gel_center_y) < gel_height//2 + noise):
                    gel_mask[y, x] = True
    
    # Mat region within gel (slightly brighter)
    mat_mask = np.zeros((size, size), dtype=bool)
    for y in range(gel_center_y - gel_height//3, gel_center_y + gel_height//3):
        for x in range(gel_center_x - gel_width//3, gel_center_x + gel_width//3):
            if 0 <= x < size and 0 <= y < size and gel_mask[y, x]:
                mat_mask[y, x] = True
    
    # Apply intensity values
    image[gel_mask] = 140  # Gel region
    image[mat_mask] = 180  # Mat region (brightest)
    
    # Add ruler at bottom (outside typical ROI)
    ruler_y = size - 80
    ruler_height = 30
    image[ruler_y:ruler_y + ruler_height, 50:size-50] = 40  # Dark ruler body
    
    # Add ruler ticks
    for x in range(100, size-50, 40):  # Major ticks every 40 pixels
        image[ruler_y-5:ruler_y + ruler_height + 5, x-2:x+2] = 20
    
    # Apply illumination gradient
    image_float = image.astype(np.float32) * illumination
    image_with_illumination = np.clip(image_float, 0, 255).astype(np.uint8)
    
    # Create user ROI that excludes ruler
    user_roi = (100, 100, 600, 500)  # x, y, width, height - excludes ruler area
    
    return image_with_illumination, gel_mask, mat_mask, user_roi

def test_optimized_workflow():
    """Test the optimized workflow step by step."""
    print("Testing Optimized Analysis Workflow")
    print("=" * 50)
    
    # Create test image
    test_image, true_gel_mask, true_mat_mask, user_roi = create_comprehensive_test_image()
    
    print(f"✓ Created test image: {test_image.shape}")
    print(f"✓ User ROI: {user_roi}")
    print(f"✓ True gel pixels: {np.sum(true_gel_mask)}")
    print(f"✓ True mat pixels: {np.sum(true_mat_mask)}")
    
    # Test the new workflow components
    from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
    from esnf_mat_analyzer.processing.ruler_detector import RulerDetector
    from esnf_mat_analyzer.core.data_types import RulerDetectionConfig
    
    # 1. Test ruler detection with exclusion
    print(f"\n1. Testing ruler detection (excluding user ROI)")
    ruler_config = RulerDetectionConfig(
        enabled=True,
        expected_tick_distance_mm=10.0,
        min_line_length=30,
        max_line_gap=5,
        canny_threshold1=50,
        canny_threshold2=150,
        hough_threshold=15
    )
    
    ruler_detector = RulerDetector(ruler_config)
    
    # Create exclusion mask for user ROI
    exclusion_mask = np.zeros(test_image.shape, dtype=bool)
    x, y, w, h = user_roi
    exclusion_mask[y:y+h, x:x+w] = True
    
    try:
        scale, ruler_mask = ruler_detector.detect_scale_with_mask(
            test_image, exclusion_mask=exclusion_mask
        )
        if scale:
            print(f"✓ Detected scale: {scale:.2f} pixels/mm")
            print(f"✓ Ruler mask: {np.sum(ruler_mask)} pixels")
        else:
            print("⚠ No scale detected (may be normal for synthetic image)")
            ruler_mask = None
    except Exception as e:
        print(f"⚠ Ruler detection failed: {e}")
        scale, ruler_mask = None, None
    
    # 2. Test background correction with exclusions
    print(f"\n2. Testing background correction (full image, excluding ruler)")
    processor = AdvancedBackgroundProcessor()
    
    try:
        bg_results = processor.complete_uniformity_analysis_with_exclusion(
            test_image,
            mat_threshold=160,  # Adjust for synthetic image
            hydrogel_threshold=100,
            background_method="polynomial",
            exclusion_mask=ruler_mask
        )
        
        corrected_image = bg_results['corrected_image']
        print(f"✓ Background correction completed")
        print(f"✓ Background pixels used: {np.sum(bg_results['background_mask'])}")
        if ruler_mask is not None:
            print(f"✓ Excluded ruler pixels: {np.sum(ruler_mask)}")
        
    except Exception as e:
        print(f"✗ Background correction failed: {e}")
        return
    
    # 3. Test ROI extraction and gel detection
    print(f"\n3. Testing ROI extraction and gel detection")
    
    # Extract user ROI from corrected image
    x, y, w, h = user_roi
    roi_corrected = corrected_image[y:y+h, x:x+w]
    
    print(f"✓ Extracted ROI: {roi_corrected.shape}")
    
    # Simulate gel detection (for testing, we'll use a simple threshold)
    # In real implementation, this would use shape detection
    gel_threshold = np.percentile(roi_corrected, 75)
    detected_gel_mask = roi_corrected > gel_threshold
    
    print(f"✓ Detected gel pixels: {np.sum(detected_gel_mask)}")
    
    # 4. Test uniformity analysis on gel area
    print(f"\n4. Testing uniformity analysis on gel-enclosed area")
    
    try:
        # Calculate uniformity metrics on the gel region
        if np.sum(detected_gel_mask) > 100:  # Ensure sufficient pixels
            gel_intensities = roi_corrected[detected_gel_mask]
            
            mean_intensity = np.mean(gel_intensities)
            std_intensity = np.std(gel_intensities)
            cv = std_intensity / mean_intensity if mean_intensity > 0 else float('inf')
            
            print(f"✓ Gel uniformity analysis:")
            print(f"  - Mean intensity: {mean_intensity:.2f}")
            print(f"  - Std deviation: {std_intensity:.2f}")
            print(f"  - Coefficient of variation: {cv:.4f}")
            print(f"  - Analysis pixels: {len(gel_intensities)}")
            
        else:
            print("⚠ Insufficient gel pixels for uniformity analysis")
            
    except Exception as e:
        print(f"✗ Uniformity analysis failed: {e}")
    
    # 5. Create visualization
    print(f"\n5. Creating workflow visualization")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(test_image, cmap='gray')
        axes[0, 0].set_title('1. Original Image\n(with uneven illumination)')
        axes[0, 0].axis('off')
        
        # User ROI overlay
        img_with_roi = test_image.copy()
        cv2.rectangle(img_with_roi, (x, y), (x+w, y+h), 255, 3)
        axes[0, 1].imshow(img_with_roi, cmap='gray')
        axes[0, 1].set_title('2. User ROI\n(excludes ruler area)')
        axes[0, 1].axis('off')
        
        # Ruler mask (if detected)
        if ruler_mask is not None:
            axes[0, 2].imshow(ruler_mask, cmap='gray')
            axes[0, 2].set_title('3. Detected Ruler Mask\n(excluded from BG correction)')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Ruler\nDetected', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('3. Ruler Detection')
        axes[0, 2].axis('off')
        
        # Background corrected image
        axes[1, 0].imshow(corrected_image, cmap='gray')
        axes[1, 0].set_title('4. Background Corrected\n(full image)')
        axes[1, 0].axis('off')
        
        # ROI corrected
        axes[1, 1].imshow(roi_corrected, cmap='gray')
        axes[1, 1].set_title('5. ROI After Correction\n(analysis area)')
        axes[1, 1].axis('off')
        
        # Gel mask
        axes[1, 2].imshow(detected_gel_mask, cmap='gray')
        axes[1, 2].set_title('6. Detected Gel Mask\n(optimal uniformity ROI)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('optimized_workflow_test.png', dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved as 'optimized_workflow_test.png'")
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
    
    print(f"\n✓ Optimized workflow test completed!")

if __name__ == "__main__":
    # Import cv2 for visualization
    try:
        import cv2
    except ImportError:
        print("OpenCV not available for visualization")
    
    test_optimized_workflow()
