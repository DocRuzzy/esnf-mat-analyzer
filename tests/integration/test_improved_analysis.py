#!/usr/bin/env python3
"""
Fix for multiscale uniformity and ROI bounds issues.
This script addresses both the perfect uniformity scores and the ROI bounds validation.
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
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_improved_analysis():
    """Test the analysis with improved ROI handling and background correction."""
    
    print("=== IMPROVED MULTISCALE ANALYSIS TEST ===\n")
    
    # Use the same image from your test
    image_path = Path('tests/Samples/square.png')
    if not image_path.exists():
        print("Test image not found!")
        return
        
    # Load image to check dimensions
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Test with different background correction methods
    background_methods = [
        ("none", "No background correction"),
        ("basic", "Basic background correction"),
        ("rolling_ball", "Rolling ball background correction"),
        ("restore", "RESTORE background correction"),
    ]
    
    # Create a more conservative ROI that focuses on the mat center
    # Avoid edges and potential ruler areas
    margin_x = w // 6  # 16.7% margin from edges
    margin_y = h // 6  # 16.7% margin from edges
    safe_roi = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    print(f"Using safe ROI: {safe_roi}")
    print(f"ROI covers central 66% of image (avoiding edges and potential rulers)\n")
    
    for bg_method, description in background_methods:
        print(f"--- Testing with {description} ---")
        
        # Setup analyzer with specific background method
        config = get_default_config()
        
        # Set background correction method
        if bg_method == "none":
            config.processing.background_correction_method = config.processing.background_correction_method.__class__(1)  # NONE
        elif bg_method == "basic":
            config.processing.background_correction_method = config.processing.background_correction_method.__class__(2)  # BASIC
        elif bg_method == "rolling_ball":
            config.processing.background_correction_method = config.processing.background_correction_method.__class__(3)  # ROLLING_BALL
        elif bg_method == "restore":
            config.processing.background_correction_method = config.processing.background_correction_method.__class__(4)  # RESTORE
        
        analyzer = setup_dependencies(config)
        
        try:
            # Run analysis
            result = analyzer.process_image(image_path, roi=safe_roi)
            
            print(f"Processing time: {result.processing_time:.2f}s")
            
            # Check multiscale results specifically
            scale_metrics = {}
            for name, value in result.metrics.items():
                if name.startswith('scale_') and name.endswith('_uniformity'):
                    scale_level = name.replace('scale_', '').replace('_uniformity', '')
                    scale_metrics[int(scale_level)] = value
            
            if scale_metrics:
                print("Multiscale uniformity results:")
                perfect_scores = 0
                for scale in sorted(scale_metrics.keys()):
                    value = scale_metrics[scale]
                    print(f"  Scale {scale}: {value:.6f}")
                    if abs(value - 1.0) < 1e-6:
                        perfect_scores += 1
                
                overall_score = np.mean(list(scale_metrics.values()))
                print(f"  Overall: {overall_score:.6f}")
                
                if perfect_scores == len(scale_metrics):
                    print("  🚨 ALL SCALES RETURNED PERFECT SCORES!")
                elif perfect_scores > 0:
                    print(f"  ⚠️  {perfect_scores}/{len(scale_metrics)} scales returned perfect scores")
                else:
                    print("  ✅ Realistic uniformity scores obtained")
            else:
                print("  ❌ No multiscale metrics found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_improved_analysis()
