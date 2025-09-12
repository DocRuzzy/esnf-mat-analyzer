#!/usr/bin/env python3
"""
Debug the mat detection issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor

def debug_mat_detection():
    """Debug why the mat pixels aren't being detected."""
    # Create simple test image
    size = 400
    center = size // 2
    
    # Create circular mat region
    y, x = np.ogrid[:size, :size]
    mat_mask = (x - center)**2 + (y - center)**2 < (size//3)**2
    
    # Create base image
    image = np.ones((size, size), dtype=np.uint8) * 100  # Background
    image[mat_mask] = 180  # Mat region
    
    print(f"Image statistics:")
    print(f"  Min: {image.min()}")
    print(f"  Max: {image.max()}")
    print(f"  Mean: {image.mean():.1f}")
    print(f"  Mat pixels (bright): {np.sum(image > 160)}")
    print(f"  Background pixels: {np.sum(image <= 160)}")
    
    # Test the processor
    processor = AdvancedBackgroundProcessor()
    
    # Test Step 1 mask creation with different thresholds
    print(f"\nTesting mask creation:")
    
    # Try different thresholds
    for mat_thresh in [160, 170, 180, 190]:
        masks = processor._step1_isolate_background_pixels(image, mat_thresh, 50)
        mat_pixels = np.sum(masks['mat_mask'])
        bg_pixels = np.sum(masks['background_mask'])
        print(f"  Mat threshold {mat_thresh}: {mat_pixels} mat pixels, {bg_pixels} bg pixels")
    
    # Test with lower threshold that should work
    print(f"\nTesting complete workflow with mat_threshold=160:")
    results = processor.complete_uniformity_analysis(
        image, 
        mat_threshold=160,
        hydrogel_threshold=50,
        background_method="polynomial"
    )
    
    print(f"CV: {results['coefficient_of_variation']:.4f}")
    print(f"Mean intensity: {results['mean_intensity']:.1f}")
    print(f"Mat pixels: {results['mat_pixels']}")
    
    return results

if __name__ == "__main__":
    debug_mat_detection()
