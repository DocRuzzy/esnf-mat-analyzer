#!/usr/bin/env python3
"""
Test to reproduce the exact conditions that lead to perfect uniformity scores.
This script will test with different ROI sizes and positions to find the issue.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import logging
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_different_roi_conditions():
    """Test multiscale analysis with different ROI conditions to find the perfect scores issue."""
    
    print("=== TESTING DIFFERENT ROI CONDITIONS ===\n")
    
    # Load test image
    image_path = Path('tests/Samples/square.png')
    if not image_path.exists():
        print("Test image not found!")
        return
        
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {image.shape}")
    
    thickness_map = 255 - image.astype(np.float32)
    h, w = image.shape
    
    # Test different ROI conditions
    test_cases = [
        ("Small ROI (10x10)", (w//2-5, h//2-5, 10, 10)),
        ("Medium ROI (100x100)", (w//2-50, h//2-50, 100, 100)),
        ("Large ROI (500x500)", (w//2-250, h//2-250, 500, 500)),
        ("Very uniform region (ruler area)", (100, 2000, 200, 200)),
        ("Mixed region (center)", (w//4, h//4, w//2, h//2)),
        ("Edge region", (0, 0, 200, 200)),
    ]
    
    analyzer = MultiScaleUniformityAnalyzer()
    
    for name, roi in test_cases:
        print(f"\n--- {name} ---")
        x, y, width, height = roi
        
        # Check bounds
        if x + width > w or y + height > h or x < 0 or y < 0:
            print(f"ROI out of bounds, skipping: {roi}")
            continue
            
        # Create ROI mask
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y:y+height, x:x+width] = True
        
        roi_pixels = np.sum(roi_mask)
        print(f"ROI: x={x}, y={y}, w={width}, h={height} ({roi_pixels} pixels)")
        
        # Get thickness values in ROI
        roi_thickness = thickness_map[roi_mask]
        if len(roi_thickness) > 0:
            print(f"ROI thickness: mean={roi_thickness.mean():.2f}, std={roi_thickness.std():.2f}")
            print(f"ROI range: {roi_thickness.min():.2f} to {roi_thickness.max():.2f}")
            print(f"Unique values in ROI: {len(np.unique(roi_thickness))}")
            
            # Check for very uniform regions
            if roi_thickness.std() < 1e-6:
                print("⚠️  ROI is extremely uniform (std < 1e-6)")
            if len(np.unique(roi_thickness)) < 5:
                print("⚠️  ROI has very few unique values")
        
        # Run multiscale analysis
        try:
            results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
            print("Results:")
            
            perfect_scores = 0
            for key, value in results.items():
                print(f"  {key}: {value:.6f}")
                if abs(value - 1.0) < 1e-6:
                    perfect_scores += 1
                    
            if perfect_scores == len(results):
                print("🚨 ALL SCALES RETURNED PERFECT SCORES!")
            elif perfect_scores > 0:
                print(f"⚠️  {perfect_scores}/{len(results)} scales returned perfect scores")
                
        except Exception as e:
            print(f"Error in multiscale analysis: {e}")
    
    # Test with synthetic uniform data
    print(f"\n--- Synthetic Uniform Data Test ---")
    uniform_data = np.ones((100, 100), dtype=np.float32) * 128.0
    uniform_mask = np.ones((100, 100), dtype=bool)
    
    print("Testing with perfectly uniform synthetic data...")
    try:
        results = analyzer.analyze_multiscale_uniformity(uniform_data, uniform_mask)
        print("Results for uniform data:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
    except Exception as e:
        print(f"Error with uniform data: {e}")
    
    # Test with synthetic noisy data
    print(f"\n--- Synthetic Noisy Data Test ---")
    np.random.seed(42)  # For reproducible results
    noisy_data = np.random.normal(128.0, 10.0, (100, 100)).astype(np.float32)
    noisy_mask = np.ones((100, 100), dtype=bool)
    
    print("Testing with noisy synthetic data (mean=128, std=10)...")
    try:
        results = analyzer.analyze_multiscale_uniformity(noisy_data, noisy_mask)
        print("Results for noisy data:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
    except Exception as e:
        print(f"Error with noisy data: {e}")

if __name__ == "__main__":
    test_different_roi_conditions()
