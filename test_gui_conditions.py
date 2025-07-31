#!/usr/bin/env python3
"""
Test script to reproduce the exact GUI analysis conditions.
This script simulates the full pipeline that runs in the GUI.
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
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.config.config_manager import get_default_config
from esnf_mat_analyzer.main import setup_dependencies

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gui_conditions():
    """Test the same conditions as the GUI with the exact same image and ROI."""
    
    print("=== GUI CONDITIONS TEST ===\n")
    
    # Use the same image from your test
    image_path = Path('tests/Samples/square.png')
    if not image_path.exists():
        print("Test image not found!")
        return
        
    # Load image to check dimensions
    image = cv2.imread(str(image_path))
    print(f"Image dimensions: {image.shape}")
    
    # The ROI from your error message: 655,828,1310,1470
    # This means: x=655, y=828, width=1310, height=1470
    # But the error says this extends beyond bounds for image 1816x2420
    # Let's check what ROI would be reasonable
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Test with the exact ROI from your error message: 655,828,1310,1470
    # But first make sure it fits in image bounds
    roi_x, roi_y, roi_width, roi_height = 655, 828, 1310, 1470
    
    # Check if this ROI extends beyond bounds (like your error)
    if roi_x + roi_width > w or roi_y + roi_height > h:
        print(f"Original ROI extends beyond bounds: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
        print(f"Image bounds: {w}x{h}")
        
        # Clamp the ROI to fit
        roi_width = min(roi_width, w - roi_x)
        roi_height = min(roi_height, h - roi_y)
        print(f"Clamped ROI: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
    
    roi = (roi_x, roi_y, roi_width, roi_height)
    
    print(f"Safe ROI: {roi}")
    print(f"ROI bounds: x={roi_x} to {roi_x + roi_width}, y={roi_y} to {roi_y + roi_height}")
    
    # Setup analyzer exactly like in GUI
    config = get_default_config()
    analyzer = setup_dependencies(config)
    
    print(f"\nBackground method: {config.processing.background_correction_method}")
    print(f"Thickness method: {config.thickness.model_type}")
    
    try:
        # Run the full analysis pipeline
        result = analyzer.process_image(image_path, roi=roi)
        
        print(f"\nProcessing completed in {result.processing_time:.2f}s")
        print(f"Spatial scale: {result.spatial_scale_pixels_per_mm:.2f} pixels/mm")
        
        print(f"\nThickness map shape: {result.thickness_map.shape}")
        print(f"Thickness range: {result.thickness_map.min():.2f} to {result.thickness_map.max():.2f}")
        
        # Check mask
        valid_pixels = np.sum(result.mask > 0)
        print(f"Valid pixels in mask: {valid_pixels}")
        
        # Display metrics exactly like GUI does
        print(f"\nTraditional Uniformity Metrics:")
        for name, value in result.metrics.items():
            if not name.startswith('scale_'):
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        
        print(f"\nMat-Scale Uniformity Analysis:")
        print("=" * 40)
        print("Multiscale Uniformity Analysis:")
        print("-" * 35)
        
        scale_metrics = {}
        for name, value in result.metrics.items():
            if name.startswith('scale_') and name.endswith('_uniformity'):
                scale_level = name.replace('scale_', '').replace('_uniformity', '')
                scale_metrics[int(scale_level)] = value
        
        overall_score = 0.0
        valid_scales = 0
        
        for scale in sorted(scale_metrics.keys()):
            value = scale_metrics[scale]
            print(f"  Scale Level {scale}: {value:.4f}")
            if not np.isnan(value):
                overall_score += value
                valid_scales += 1
        
        if valid_scales > 0:
            overall_score /= valid_scales
            print(f"\n  Overall Multiscale Uniformity: {overall_score:.4f}")
            
            # Rating
            if overall_score >= 0.9:
                rating = "Excellent"
            elif overall_score >= 0.8:
                rating = "Good"
            elif overall_score >= 0.7:
                rating = "Fair"
            elif overall_score >= 0.6:
                rating = "Poor"
            else:
                rating = "Very Poor"
            print(f"  Multiscale Rating: {rating}")
        else:
            print(f"\n  Overall Multiscale Uniformity: 0.0000")
            print(f"  Multiscale Rating: No valid data")
        
        # Debug: Check if there's an issue with the actual analysis
        print(f"\n=== Debug Info ===")
        print(f"Raw metrics dump:")
        for name, value in result.metrics.items():
            if name.startswith('scale_'):
                print(f"  {name}: {value} (type: {type(value)})")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_conditions()
