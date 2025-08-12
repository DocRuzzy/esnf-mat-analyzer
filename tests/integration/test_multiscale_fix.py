#!/usr/bin/env python3
"""
Test script for the multiscale uniformity analysis improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer

def create_test_data():
    """Create synthetic test data with known patterns."""
    # Create a 256x256 test image
    size = 256
    thickness_map = np.zeros((size, size))
    
    # Add various patterns at different scales
    
    # Large scale gradient (scale 0)
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    thickness_map += 0.5 * x  # Linear gradient
    
    # Medium scale features (scale 1-2)  
    for i in range(5):
        for j in range(5):
            cx, cy = i * size // 5 + size // 10, j * size // 5 + size // 10
            thickness_map[cx-20:cx+20, cy-20:cy+20] += 0.2 * np.sin(np.linspace(0, 2*np.pi, 40))[:, None]
    
    # Fine scale noise (scale 3+)
    thickness_map += 0.1 * np.random.random((size, size))
    
    # Create ROI mask (circular region)
    center = size // 2
    radius = size // 3
    y_coords, x_coords = np.ogrid[:size, :size]
    distance = np.sqrt((x_coords - center)**2 + (y_coords - center)**2)
    roi_mask = distance <= radius
    
    return thickness_map, roi_mask

def test_multiscale_analysis():
    """Test the multiscale uniformity analysis."""
    print("Creating test data...")
    thickness_map, roi_mask = create_test_data()
    
    print(f"Test data shapes: thickness_map {thickness_map.shape}, roi_mask {roi_mask.shape}")
    print(f"ROI coverage: {np.sum(roi_mask) / roi_mask.size * 100:.1f}%")
    print(f"Thickness range: {np.min(thickness_map):.3f} to {np.max(thickness_map):.3f}")
    
    # Initialize analyzer
    analyzer = MultiScaleUniformityAnalyzer(wavelet='db4', levels=4)
    
    # Run analysis
    print("\nRunning multiscale uniformity analysis...")
    results = analyzer.analyze_multiscale_uniformity(thickness_map, roi_mask)
    
    print("\nResults:")
    for key, value in sorted(results.items()):
        level = key.replace('scale_', '').replace('_uniformity', '')
        print(f"  Scale Level {level}: {value:.6f}")
    
    # Calculate overall uniformity
    valid_values = [v for v in results.values() if isinstance(v, float) and not np.isnan(v) and v > 0]
    if valid_values:
        overall_uniformity = np.mean(valid_values)
        print(f"\n  Overall Multiscale Uniformity: {overall_uniformity:.6f}")
        
        # Rating system
        if overall_uniformity >= 0.8:
            rating = "Excellent"
        elif overall_uniformity >= 0.6:
            rating = "Good"
        elif overall_uniformity >= 0.4:
            rating = "Fair"
        elif overall_uniformity >= 0.2:
            rating = "Poor"
        else:
            rating = "Very Poor"
        print(f"  Multiscale Rating: {rating}")
    else:
        print("\n  No valid uniformity values found!")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original thickness map
    im1 = axes[0].imshow(thickness_map, cmap='viridis')
    axes[0].set_title('Original Thickness Map')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # ROI mask
    axes[1].imshow(roi_mask, cmap='gray')
    axes[1].set_title('ROI Mask')
    axes[1].axis('off')
    
    # Masked thickness map (what analysis sees)
    masked_thickness = np.ma.masked_array(thickness_map, mask=~roi_mask)
    im3 = axes[2].imshow(masked_thickness, cmap='viridis')
    axes[2].set_title('Masked Thickness Map')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('multiscale_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: multiscale_test_results.png")
    
    return results

if __name__ == "__main__":
    try:
        results = test_multiscale_analysis()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
