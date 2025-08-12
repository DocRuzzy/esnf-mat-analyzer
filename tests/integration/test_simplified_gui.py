#!/usr/bin/env python3
"""
Test script for the simplified GUI background correction interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor

def create_test_image():
    """Create a synthetic test image with uneven illumination."""
    # Create base mat structure
    size = 400
    center = size // 2
    
    # Create circular mat region
    y, x = np.ogrid[:size, :size]
    mat_mask = (x - center)**2 + (y - center)**2 < (size//3)**2
    
    # Create base image
    image = np.ones((size, size), dtype=np.uint8) * 100  # Background
    image[mat_mask] = 180  # Mat region
    
    # Add uneven illumination (gradient)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    illumination = 0.5 + 0.5 * (xx + 0.3 * yy)  # Diagonal gradient
    
    # Apply illumination
    image_float = image.astype(np.float32) * illumination
    image_with_illumination = np.clip(image_float, 0, 255).astype(np.uint8)
    
    return image_with_illumination

def test_simplified_workflow():
    """Test the simplified workflow that always uses 4 steps."""
    print("Testing Simplified Background Correction Workflow")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize processor
    processor = AdvancedBackgroundProcessor()
    
    # Test Method 2A (Polynomial)
    print("\nTesting Method 2A: Polynomial Surface")
    results_2a = processor.complete_uniformity_analysis(
        test_image, 
        mat_threshold=160,  # Lower threshold for synthetic image
        hydrogel_threshold=50,
        background_method="polynomial"
    )
    
    print(f"✓ CV (Polynomial): {results_2a['coefficient_of_variation']:.4f}")
    print(f"✓ Mean intensity: {results_2a['mean_intensity']:.1f}")
    print(f"✓ Mat pixels: {results_2a['mat_pixels']}")
    
    # Test Method 2B (Large Kernel Blur)
    print("\nTesting Method 2B: Large Kernel Blur")
    results_2b = processor.complete_uniformity_analysis(
        test_image, 
        mat_threshold=160,  # Lower threshold for synthetic image
        hydrogel_threshold=50,
        background_method="large_kernel_blur"
    )
    
    print(f"✓ CV (Large Blur): {results_2b['coefficient_of_variation']:.4f}")
    print(f"✓ Mean intensity: {results_2b['mean_intensity']:.1f}")
    print(f"✓ Mat pixels: {results_2b['mat_pixels']}")
    
    # Compare results
    print("\nComparison:")
    print(f"CV difference: {abs(results_2a['coefficient_of_variation'] - results_2b['coefficient_of_variation']):.4f}")
    print(f"Method 2A CV: {results_2a['coefficient_of_variation']:.4f}")
    print(f"Method 2B CV: {results_2b['coefficient_of_variation']:.4f}")
    
    if results_2a['coefficient_of_variation'] < results_2b['coefficient_of_variation']:
        print("→ Method 2A (Polynomial) gives lower CV")
    else:
        print("→ Method 2B (Large Blur) gives lower CV")
    
    print("\n✓ All tests completed successfully!")
    return results_2a, results_2b

if __name__ == "__main__":
    test_simplified_workflow()
