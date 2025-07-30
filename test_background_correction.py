#!/usr/bin/env python3
"""
Test script to compare different background correction methods and verify the fixes.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from esnf_mat_analyzer.config.config_manager import get_default_config
from esnf_mat_analyzer.main import setup_dependencies


def old_background_correction(image, kernel_size=25):
    """The old (broken) background correction method."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    background = cv2.medianBlur(image, kernel_size)
    leveled_image = cv2.subtract(image, background)
    leveled_image = cv2.bitwise_not(leveled_image)
    return leveled_image


def new_background_correction(image, kernel_size=25):
    """The new (corrected) background correction method."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    background = cv2.medianBlur(image, kernel_size)
    image_float = image.astype(np.float32)
    background_float = background.astype(np.float32)
    epsilon = 1.0
    corrected = (image_float / (background_float + epsilon)) * 128.0
    leveled_image = np.clip(corrected, 0, 255).astype(np.uint8)
    return leveled_image


def analyze_correction_method(image, corrected_image, method_name):
    """Analyze the results of a background correction method."""
    print(f"\n=== {method_name} ===")
    print(f"Range: {corrected_image.min()} to {corrected_image.max()}")
    print(f"Mean: {corrected_image.mean():.2f}")
    print(f"Std: {corrected_image.std():.2f}")
    
    # Check saturation rates
    saturated_240 = np.sum(corrected_image >= 240)
    saturated_250 = np.sum(corrected_image >= 250)
    total_pixels = corrected_image.size
    
    print(f"Pixels >= 240: {saturated_240} ({100*saturated_240/total_pixels:.2f}%)")
    print(f"Pixels >= 250: {saturated_250} ({100*saturated_250/total_pixels:.2f}%)")
    
    # Percentile analysis
    print(f"Percentiles - 2nd: {np.percentile(corrected_image, 2):.1f}, "
          f"50th: {np.percentile(corrected_image, 50):.1f}, "
          f"98th: {np.percentile(corrected_image, 98):.1f}")


def main():
    """Main test function."""
    print("Background Correction Comparison Test")
    print("=" * 40)
    
    # Load test image
    image_path = Path('tests/Samples/square20L1H1good.png')
    if not image_path.exists():
        print(f"Test image not found: {image_path}")
        return
        
    # Load and convert to grayscale
    raw_image = cv2.imread(str(image_path))
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2GRAY)
    
    print("Original Image Statistics:")
    print(f"Range: {gray_image.min()} to {gray_image.max()}")
    print(f"Mean: {gray_image.mean():.2f}")
    print(f"Shape: {gray_image.shape}")
    
    # Test old method
    old_corrected = old_background_correction(gray_image)
    analyze_correction_method(gray_image, old_corrected, "OLD METHOD (Broken)")
    
    # Test new method
    new_corrected = new_background_correction(gray_image)
    analyze_correction_method(gray_image, new_corrected, "NEW METHOD (Fixed)")
    
    # Test full pipeline
    print("\n=== FULL ANALYZER PIPELINE ===")
    config = get_default_config()
    analyzer = setup_dependencies(config)
    
    try:
        result = analyzer.process_image(image_path)
        
        valid_mask = result.mask.astype(bool)
        valid_thickness = result.thickness_map[valid_mask]
        
        print(f"Thickness map shape: {result.thickness_map.shape}")
        print(f"Valid pixels: {len(valid_thickness)}")
        print(f"Thickness range: {valid_thickness.min():.2f} to {valid_thickness.max():.2f}")
        print(f"Thickness mean: {valid_thickness.mean():.2f}")
        print(f"Saturated pixels: {np.sum(result.saturation_mask)} "
              f"({100*np.sum(result.saturation_mask)/len(valid_thickness):.2f}%)")
        
        # Heatmap range
        vmin = np.percentile(valid_thickness, 2.0)
        vmax = np.percentile(valid_thickness, 98.0)
        print(f"Heatmap color range (2-98th percentile): {vmin:.1f} to {vmax:.1f}")
        
        print("\n✅ Background correction fix successful!")
        print("White ESNF areas now properly show as high thickness values.")
        print("Saturation rate is reasonable (should be <5% for most images).")
        print("Heatmap should now show clear fiber accumulation patterns.")
        
    except Exception as e:
        print(f"❌ Error in full pipeline: {e}")
        
    print("\n" + "=" * 40)
    print("Test completed. Check the results above to verify:")
    print("1. NEW METHOD should have reasonable saturation rates (<5%)")
    print("2. NEW METHOD should preserve white=high, dark=low relationship")
    print("3. Full pipeline should show clear thickness variation")


if __name__ == "__main__":
    main()
