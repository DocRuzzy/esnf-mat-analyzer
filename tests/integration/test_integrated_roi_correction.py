#!/usr/bin/env python3
"""
Test the integrated ROI-aware background correction methods in the main pipeline.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
import logging

def test_integrated_roi_aware_correction():
    """Test the integrated ROI-aware background correction methods."""
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    print("Testing Integrated ROI-Aware Background Correction")
    print("=" * 60)
    
    # Load test image
    test_image_path = Path("tests/Samples/square.png")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
        print(f"✓ Loaded test image: {test_image_path}")
    else:
        # Create synthetic test image with gradient and texture
        h, w = 400, 600
        image = np.ones((h, w), dtype=np.uint8) * 120
        
        # Add illumination gradient
        y, x = np.ogrid[:h, :w]
        gradient = 40 * (1 - y / h) + 20 * (1 - x / w)
        image = np.clip(image.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
        
        # Add texture in center region (simulating gel mat)
        center_y, center_x = h // 2, w // 2
        roi_h, roi_w = h // 2, w // 2
        texture = np.random.normal(0, 15, (roi_h, roi_w))
        
        y_start = center_y - roi_h // 2
        y_end = y_start + roi_h
        x_start = center_x - roi_w // 2
        x_end = x_start + roi_w
        
        image[y_start:y_end, x_start:x_end] = np.clip(
            image[y_start:y_end, x_start:x_end].astype(np.float32) + texture, 0, 255
        ).astype(np.uint8)
        
        print("✓ Created synthetic test image with gradient and texture")
    
    # Create ROI mask (center region representing the gel square)
    h, w = image.shape
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_h = min(h // 2, 300)
    roi_w = min(w // 2, 300)
    center_y, center_x = h // 2, w // 2
    
    y_start = center_y - roi_h // 2
    y_end = y_start + roi_h
    x_start = center_x - roi_w // 2
    x_end = x_start + roi_w
    
    roi_mask[y_start:y_end, x_start:x_end] = True
    
    print(f"✓ Created ROI mask: {np.sum(roi_mask)} pixels ({np.sum(roi_mask)/roi_mask.size*100:.1f}%)")
    
    # Initialize processors
    bg_processor = AdvancedBackgroundProcessor()
    uniformity_analyzer = MultiScaleUniformityAnalyzer()
    
    # Test all new ROI-aware methods
    methods = [
        "gel_square",
        "restore", 
        "adaptive_rolling",
        "enhanced_percentile"
    ]
    
    print(f"\nOriginal image stats:")
    print(f"  Range: {image.min()}-{image.max()}")
    print(f"  Mean: {np.mean(image):.2f}")
    print(f"  ROI mean: {np.mean(image[roi_mask]):.2f}")
    print(f"  ROI std: {np.std(image[roi_mask]):.2f}")
    
    results = {}
    uniformity_scores = {}
    
    print(f"\nTesting ROI-aware background correction methods:")
    print("-" * 50)
    
    for method in methods:
        try:
            # Apply background correction
            corrected = bg_processor.roi_aware_background_correction(
                image, roi_mask, method=method
            )
            
            # Analyze corrected image
            roi_mean = np.mean(corrected[roi_mask])
            roi_std = np.std(corrected[roi_mask])
            
            # Calculate multiscale uniformity on corrected image
            uniformity_result = uniformity_analyzer.analyze_multiscale_uniformity(
                corrected, roi_mask
            )
            uniformity_scores[method] = uniformity_result
            
            results[method] = corrected
            
            # Extract individual scale scores
            scale_scores = []
            scales = []
            for key, value in uniformity_result.items():
                if key.endswith('_uniformity'):
                    scale_scores.append(value)
                    scale_num = key.split('_')[1]
                    scales.append(scale_num)
            
            overall_score = np.mean(scale_scores) if scale_scores else 0.0
            
            print(f"✓ {method.title().replace('_', ' ')}:")
            print(f"    Range: {corrected.min()}-{corrected.max()}")
            print(f"    ROI mean: {roi_mean:.2f} (±{roi_std:.2f})")
            print(f"    Uniformity scores: {[f'{s:.3f}' for s in scale_scores]}")
            print(f"    Overall uniformity: {overall_score:.3f}")
            
        except Exception as e:
            print(f"✗ {method.title().replace('_', ' ')}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with old methods
    print(f"\nComparing with legacy methods:")
    print("-" * 40)
    
    legacy_methods = [
        ("Rolling Ball (Legacy)", lambda: bg_processor.rolling_ball_3d(image, 50)),
        ("RESTORE (Legacy)", lambda: bg_processor.restore_method(image)),
        ("Homomorphic (Legacy)", lambda: bg_processor.homomorphic_filter(image)),
    ]
    
    for name, method_func in legacy_methods:
        try:
            corrected = method_func()
            roi_mean = np.mean(corrected[roi_mask])
            roi_std = np.std(corrected[roi_mask])
            
            # Calculate uniformity on legacy corrected image
            uniformity_result = uniformity_analyzer.analyze_multiscale_uniformity(
                corrected, roi_mask
            )
            
            # Extract scale scores
            scale_scores = []
            for key, value in uniformity_result.items():
                if key.endswith('_uniformity'):
                    scale_scores.append(value)
            
            overall_score = np.mean(scale_scores) if scale_scores else 0.0
            
            print(f"✓ {name}:")
            print(f"    ROI mean: {roi_mean:.2f} (±{roi_std:.2f})")
            print(f"    Uniformity scores: {[f'{s:.3f}' for s in scale_scores]}")
            print(f"    Overall uniformity: {overall_score:.3f}")
            
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Summary and recommendations
    print(f"\nSummary and Recommendations:")
    print("=" * 40)
    
    # Find best performing method
    if uniformity_scores:
        # Calculate overall scores for each method
        method_averages = {}
        for method, scores in uniformity_scores.items():
            scale_scores = []
            for key, value in scores.items():
                if key.endswith('_uniformity'):
                    scale_scores.append(value)
            method_averages[method] = np.mean(scale_scores) if scale_scores else 0.0
        
        best_method = max(method_averages.keys(), key=lambda k: method_averages[k])
        best_score = method_averages[best_method]
        
        print(f"Best performing method: {best_method.replace('_', ' ').title()}")
        print(f"Best overall uniformity score: {best_score:.3f}")
        
        # Show detailed scale analysis for best method
        best_result = uniformity_scores[best_method]
        print(f"\nScale-by-scale analysis for {best_method}:")
        for key, value in best_result.items():
            if key.endswith('_uniformity'):
                scale_num = key.split('_')[1]
                print(f"  Scale {scale_num}: {value:.3f}")
    
    print(f"\n✓ Successfully tested {len(results)} ROI-aware methods")
    return results, uniformity_scores

if __name__ == "__main__":
    test_integrated_roi_aware_correction()
