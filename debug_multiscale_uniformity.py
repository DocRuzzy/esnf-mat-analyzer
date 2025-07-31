#!/usr/bin/env python3
"""
Debug script for multiscale uniformity analysis.
This script helps identify why all scales are returning perfect uniformity (1.0).
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
from esnf_mat_analyzer.core.data_types import UniformityConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def debug_multiscale_analysis():
    """Debug the multiscale uniformity analysis step by step."""
    
    print("=== MULTISCALE UNIFORMITY DEBUG ===\n")
    
    # Load test image
    image_path = Path('tests/Samples/square.png')
    if not image_path.exists():
        print("Test image not found!")
        return
        
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Loaded image: {image.shape}, range: {image.min()}-{image.max()}")
    
    # Create a simple thickness map (inverted for testing)
    thickness_map = 255 - image.astype(np.float32)
    print(f"Thickness map: {thickness_map.shape}, range: {thickness_map.min():.2f}-{thickness_map.max():.2f}")
    
    # Create ROI mask (center 50% of image to avoid ruler)
    h, w = image.shape
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_y1, roi_y2 = h//4, 3*h//4
    roi_x1, roi_x2 = w//4, 3*w//4
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = True
    
    roi_pixels = np.sum(roi_mask)
    print(f"ROI mask: {roi_pixels} pixels ({roi_pixels/roi_mask.size*100:.1f}% of image)")
    
    # Check thickness values in ROI
    roi_thickness = thickness_map[roi_mask]
    print(f"ROI thickness: mean={roi_thickness.mean():.2f}, std={roi_thickness.std():.2f}")
    print(f"ROI thickness range: {roi_thickness.min():.2f} to {roi_thickness.max():.2f}")
    
    # Run multiscale analysis with debug logging
    analyzer = MultiScaleUniformityAnalyzer()
    
    print("\n=== Running Multiscale Analysis ===")
    results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
    
    print("\n=== Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    
    # Manual wavelet analysis to see what's happening
    print("\n=== Manual Wavelet Debug ===")
    try:
        import pywt
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(thickness_map, 'db4', levels=4)
        
        for level in range(len(coeffs)):
            if level == 0:
                # Approximation coefficients
                coeff_array = coeffs[level]
                print(f"Scale {level} (approximation): shape={coeff_array.shape}")
                
                # Apply ROI masking
                if coeff_array.shape != roi_mask.shape:
                    downsample_factor_y = roi_mask.shape[0] // coeff_array.shape[0]
                    downsample_factor_x = roi_mask.shape[1] // coeff_array.shape[1]
                    downsampled_mask = roi_mask[::downsample_factor_y, ::downsample_factor_x]
                else:
                    downsampled_mask = roi_mask
                    
                if downsampled_mask.shape == coeff_array.shape:
                    masked_coeffs = coeff_array[downsampled_mask]
                    if len(masked_coeffs) > 0:
                        print(f"  Masked coeffs: {len(masked_coeffs)} values")
                        print(f"  Range: {masked_coeffs.min():.6f} to {masked_coeffs.max():.6f}")
                        print(f"  Mean: {masked_coeffs.mean():.6f}, Std: {masked_coeffs.std():.6f}")
                        
                        # Calculate CV manually
                        if abs(masked_coeffs.mean()) > 1e-12:
                            cv = masked_coeffs.std() / abs(masked_coeffs.mean())
                            uniformity = np.exp(-cv)
                            print(f"  CV: {cv:.6f}, Uniformity: {uniformity:.6f}")
                        else:
                            print(f"  Mean too small for CV calculation")
                    
            else:
                # Detail coefficients
                detail_coeffs = coeffs[level]
                if isinstance(detail_coeffs, tuple) and len(detail_coeffs) == 3:
                    cH, cV, cD = detail_coeffs
                    print(f"Scale {level} (details): H={cH.shape}, V={cV.shape}, D={cD.shape}")
                    
                    all_details = []
                    for i, coeff_array in enumerate([cH, cV, cD]):
                        # Apply ROI masking
                        if coeff_array.shape != roi_mask.shape:
                            downsample_factor_y = roi_mask.shape[0] // coeff_array.shape[0]
                            downsample_factor_x = roi_mask.shape[1] // coeff_array.shape[1]
                            if downsample_factor_y > 0 and downsample_factor_x > 0:
                                downsampled_mask = roi_mask[::downsample_factor_y, ::downsample_factor_x]
                            else:
                                continue
                        else:
                            downsampled_mask = roi_mask
                            
                        if downsampled_mask.shape == coeff_array.shape:
                            masked_coeffs = coeff_array[downsampled_mask]
                            if len(masked_coeffs) > 0:
                                all_details.append(masked_coeffs)
                                detail_type = ['H', 'V', 'D'][i]
                                print(f"  {detail_type}: {len(masked_coeffs)} values, range: {masked_coeffs.min():.6f} to {masked_coeffs.max():.6f}")
                    
                    if all_details:
                        combined_details = np.concatenate(all_details)
                        print(f"  Combined: {len(combined_details)} values")
                        print(f"  Range: {combined_details.min():.6f} to {combined_details.max():.6f}")
                        print(f"  Mean: {combined_details.mean():.6f}, Std: {combined_details.std():.6f}")
                        
                        # Calculate normalized std manually
                        data_range = combined_details.max() - combined_details.min()
                        if data_range > 1e-12:
                            normalized_std = combined_details.std() / data_range
                            uniformity = 1.0 / (1.0 + normalized_std)
                            print(f"  Data range: {data_range:.6f}")
                            print(f"  Normalized std: {normalized_std:.6f}")
                            print(f"  Uniformity: {uniformity:.6f}")
                        else:
                            print(f"  Zero range - perfect uniformity")
                    
    except ImportError:
        print("PyWavelets not available - using fallback method")

if __name__ == "__main__":
    debug_multiscale_analysis()
