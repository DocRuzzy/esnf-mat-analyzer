#!/usr/bin/env python3
"""
Test to check if the fallback method is being used instead of PyWavelets.
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

# Setup logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_pywt_availability():
    """Test if PyWavelets is available and working."""
    
    print("=== TESTING PYWT AVAILABILITY ===\n")
    
    try:
        import pywt
        print("✓ PyWavelets (pywt) imported successfully")
        print(f"  Version: {pywt.__version__}")
        
        # Test basic functionality
        test_data = np.random.random((64, 64))
        coeffs = pywt.wavedec2(test_data, 'db4', levels=4)
        print(f"✓ Wavelet decomposition working: {len(coeffs)} levels")
        
        return True
        
    except ImportError as e:
        print(f"✗ PyWavelets not available: {e}")
        return False
    except Exception as e:
        print(f"✗ PyWavelets error: {e}")
        return False

def test_multiscale_method_selection():
    """Test which method is actually being used."""
    
    print("\n=== TESTING MULTISCALE METHOD SELECTION ===\n")
    
    from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
    
    # Load test image
    image_path = Path('tests/Samples/square.png')
    if not image_path.exists():
        print("Test image not found!")
        return
        
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    thickness_map = 255 - image.astype(np.float32)
    
    # Create ROI mask
    h, w = image.shape
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_y1, roi_y2 = h//4, 3*h//4
    roi_x1, roi_x2 = w//4, 3*w//4
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = True
    
    # Test with default analyzer
    analyzer = MultiScaleUniformityAnalyzer()
    
    # Check if pywt is being used by patching it temporarily
    import esnf_mat_analyzer.analysis.multiscale_uniformity as mu_module
    
    # Save original state
    original_pywt = getattr(mu_module, 'pywt', None)
    
    print("Testing with PyWavelets available:")
    results_with_pywt = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
    for key, value in results_with_pywt.items():
        print(f"  {key}: {value:.6f}")
    
    # Now test by temporarily removing pywt
    try:
        mu_module.pywt = None
        print("\nTesting with PyWavelets disabled (fallback method):")
        results_fallback = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
        for key, value in results_fallback.items():
            print(f"  {key}: {value:.6f}")
            
        # Compare results
        print("\nComparison:")
        for key in results_with_pywt.keys():
            if key in results_fallback:
                diff = abs(results_with_pywt[key] - results_fallback[key])
                print(f"  {key}: diff = {diff:.6f}")
                
    finally:
        # Restore original state
        mu_module.pywt = original_pywt

if __name__ == "__main__":
    pywt_available = test_pywt_availability()
    test_multiscale_method_selection()
