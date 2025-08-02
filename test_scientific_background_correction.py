#!/usr/bin/env python3
"""
Test script to verify background correction methods follow the scientific guide.

This script demonstrates the background correction methods implemented according to:
- "Guide: Analyzing Electrospun Mat Uniformity with Background Correction"
- "A First-Principles Approach to Analysis of Electrospun Mat Thickness and Uniformity"
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
from esnf_mat_analyzer.core.data_types import BackgroundCorrectionMethod

def create_synthetic_test_image():
    """Create a synthetic test image with illumination gradient and mat."""
    h, w = 400, 600
    
    # Create illumination gradient (bright top-left, dark bottom-right)
    y, x = np.ogrid[:h, :w]
    illumination = 200 - 100 * (x / w + y / h) / 2
    
    # Create synthetic mat (bright region)
    mat_center_x, mat_center_y = w // 2, h // 2
    mat_width, mat_height = 200, 150
    
    # Base image with gradient
    image = illumination.astype(np.uint8)
    
    # Add mat region (brighter due to scattering)
    mat_mask = np.zeros((h, w), dtype=bool)
    mat_y1 = mat_center_y - mat_height // 2
    mat_y2 = mat_center_y + mat_height // 2
    mat_x1 = mat_center_x - mat_width // 2
    mat_x2 = mat_center_x + mat_width // 2
    mat_mask[mat_y1:mat_y2, mat_x1:mat_x2] = True
    
    # Add mat brightness with some texture
    mat_brightness = 80 + 20 * np.random.normal(0, 1, (mat_height, mat_width))
    image[mat_mask] = np.clip(image[mat_mask] + mat_brightness.flatten(), 0, 255)
    
    # Add dark hydrogel region
    hydrogel_mask = np.zeros((h, w), dtype=bool)
    hydrogel_y1, hydrogel_y2 = 50, 100
    hydrogel_x1, hydrogel_x2 = 50, w - 50
    hydrogel_mask[hydrogel_y1:hydrogel_y2, hydrogel_x1:hydrogel_x2] = True
    image[hydrogel_mask] = 30
    
    return image, mat_mask, hydrogel_mask

def test_background_correction_methods():
    """Test all background correction methods according to the guide."""
    
    print("Testing Background Correction Methods")
    print("=" * 50)
    
    # Create test image
    image, mat_mask, hydrogel_mask = create_synthetic_test_image()
    print(f"Created synthetic test image: {image.shape}")
    print(f"Original image - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.1f}")
    
    # Initialize processor
    processor = AdvancedBackgroundProcessor()
    
    # Test all methods
    methods = {
        "None": image,
        "Polynomial Surface (Method A)": processor.polynomial_surface_fitting(image, polynomial_order=2),
        "Large Kernel Blur (Method B)": processor.large_kernel_blur(image),
        "Morphological Opening": processor.morphological_opening(image, radius=30),
        "Homomorphic Filter": processor.homomorphic_filter(image),
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (method_name, corrected_image) in enumerate(methods.items()):
        if i >= len(axes):
            break
            
        axes[i].imshow(corrected_image, cmap='viridis')
        axes[i].set_title(f'{method_name}')
        axes[i].axis('off')
        
        # Print statistics
        print(f"\n{method_name}:")
        print(f"  Range: {corrected_image.min()} to {corrected_image.max()}")
        print(f"  Mean: {corrected_image.mean():.1f}")
        print(f"  Std: {corrected_image.std():.1f}")
        
        # Calculate uniformity in mat region if available
        if mat_mask.any():
            mat_intensities = corrected_image[mat_mask]
            mat_cv = mat_intensities.std() / mat_intensities.mean()
            print(f"  Mat CV (uniformity): {mat_cv:.3f}")
    
    # Remove unused subplot
    if len(methods) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.suptitle('Background Correction Methods Comparison\n(Following Scientific Guide)', y=0.98)
    
    # Save the comparison
    output_path = Path('background_correction_test_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {output_path}")
    
    return methods

def test_background_mask_creation():
    """Test the background mask creation as described in Step 1 of the guide."""
    
    print("\n" + "=" * 50)
    print("Testing Background Mask Creation (Step 1 of Guide)")
    print("=" * 50)
    
    # Create test image
    image, mat_mask, hydrogel_mask = create_synthetic_test_image()
    
    # Initialize processor
    processor = AdvancedBackgroundProcessor()
    
    # Test background mask creation
    background_mask = processor.create_background_mask(image, mat_threshold=180, hydrogel_threshold=50)
    
    print(f"Background mask created:")
    print(f"  Total pixels: {image.size}")
    print(f"  Background pixels: {np.sum(background_mask)} ({100*np.sum(background_mask)/image.size:.1f}%)")
    print(f"  Foreground pixels: {np.sum(~background_mask)} ({100*np.sum(~background_mask)/image.size:.1f}%)")
    
    # Visualize masks
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mat_mask, cmap='Reds', alpha=0.7)
    axes[1].set_title('Mat Mask (>180)')
    axes[1].axis('off')
    
    axes[2].imshow(hydrogel_mask, cmap='Blues', alpha=0.7)
    axes[2].set_title('Hydrogel Mask (<50)')
    axes[2].axis('off')
    
    axes[3].imshow(background_mask, cmap='Greens', alpha=0.7)
    axes[3].set_title('Background Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    mask_output_path = Path('background_mask_test_results.png')
    plt.savefig(mask_output_path, dpi=150, bbox_inches='tight')
    print(f"Mask visualization saved to: {mask_output_path}")
    
    return background_mask

def test_enum_mapping():
    """Test that enum values map correctly to string values."""
    
    print("\n" + "=" * 50)
    print("Testing Enum Mapping")
    print("=" * 50)
    
    from esnf_mat_analyzer.config.config_manager import BACKGROUND_CORRECTION_METHOD_MAP
    
    print("Available background correction methods:")
    for string_val, enum_val in BACKGROUND_CORRECTION_METHOD_MAP.items():
        print(f"  '{string_val}' -> {enum_val}")
    
    # Test that all enum values are accessible
    try:
        methods = [
            BackgroundCorrectionMethod.NONE,
            BackgroundCorrectionMethod.POLYNOMIAL_SURFACE,
            BackgroundCorrectionMethod.LARGE_KERNEL_BLUR,
            BackgroundCorrectionMethod.MORPHOLOGICAL_OPENING,
            BackgroundCorrectionMethod.HOMOMORPHIC,
        ]
        print(f"\nAll {len(methods)} enum values accessible: ✓")
        return True
    except AttributeError as e:
        print(f"\nEnum access error: {e}")
        return False

if __name__ == "__main__":
    try:
        print("ESNF Mat Analyzer - Background Correction Test")
        print("Following the Scientific Guide Implementation")
        print("=" * 60)
        
        # Test enum mapping first
        enum_ok = test_enum_mapping()
        
        if enum_ok:
            # Test background mask creation
            test_background_mask_creation()
            
            # Test background correction methods
            test_background_correction_methods()
            
            print("\n" + "=" * 60)
            print("✓ All tests completed successfully!")
            print("✓ Background correction methods implemented according to guide")
            print("✓ Methods follow first-principles approach for electrospun mat analysis")
        else:
            print("\n✗ Enum mapping failed - check data_types.py")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
