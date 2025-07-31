#!/usr/bin/env python3
"""
Test script to demonstrate all available background correction methods.
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
from esnf_mat_analyzer.core.data_types import BackgroundCorrectionMethod
from esnf_mat_analyzer.processing.image_processor import ImageProcessor
from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor


def test_background_method(image, method_name, processor_func, *args):
    """Test a specific background correction method."""
    print(f"\n=== Testing {method_name} ===")
    
    try:
        corrected = processor_func(image, *args)
        
        print(f"Input range: {image.min()}-{image.max()}, mean: {image.mean():.1f}")
        print(f"Output range: {corrected.min()}-{corrected.max()}, mean: {corrected.mean():.1f}")
        
        # Check saturation
        saturated = np.sum(corrected >= 240)
        total = corrected.size
        print(f"Saturation rate: {100*saturated/total:.2f}%")
        
        # Check percentiles
        p2, p50, p98 = np.percentile(corrected, [2, 50, 98])
        print(f"Percentiles (2nd, 50th, 98th): {p2:.1f}, {p50:.1f}, {p98:.1f}")
        
        return corrected
        
    except Exception as e:
        print(f"❌ Error with {method_name}: {e}")
        return None


def main():
    """Test all background correction methods."""
    print("Background Correction Methods Comparison")
    print("=" * 50)
    
    # Load test image
    image_path = Path('tests/Samples/square20L1H1good.png')
    if not image_path.exists():
        print(f"Test image not found: {image_path}")
        return
        
    # Load and convert to grayscale
    raw_image = cv2.imread(str(image_path))
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2GRAY)
    
    print(f"Original Image: {gray_image.shape}, range: {gray_image.min()}-{gray_image.max()}")
    
    # Create advanced background processor
    advanced_processor = AdvancedBackgroundProcessor()
    
    # Test all methods
    methods = []
    
    # 1. Default normalization method (current default)
    print(f"\n{'='*50}")
    print("1. NORMALIZATION METHOD (Default)")
    print("   - Divides by median-blurred background")
    print("   - Preserves white=high, dark=low relationship")
    
    kernel_size = 25
    background = cv2.medianBlur(gray_image, kernel_size)
    image_float = gray_image.astype(np.float32)
    background_float = background.astype(np.float32)
    epsilon = 1.0
    corrected = (image_float / (background_float + epsilon)) * 128.0
    default_corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    print(f"Output range: {default_corrected.min()}-{default_corrected.max()}, mean: {default_corrected.mean():.1f}")
    saturated = np.sum(default_corrected >= 240)
    print(f"Saturation rate: {100*saturated/default_corrected.size:.2f}%")
    methods.append(("Normalization (Default)", default_corrected))
    
    # 2. BASIC method (BaSiC - NMF)
    print(f"\n{'='*50}")
    print("2. BASIC METHOD (BaSiC - Non-negative Matrix Factorization)")
    print("   - Uses machine learning to separate background/foreground")
    print("   - Good for complex illumination patterns")
    
    basic_corrected = test_background_method(
        gray_image, "BASIC", advanced_processor.basic_correction, 1
    )
    if basic_corrected is not None:
        methods.append(("BASIC (NMF)", basic_corrected))
    
    # 3. Rolling Ball method
    print(f"\n{'='*50}")
    print("3. ROLLING_BALL METHOD")
    print("   - Simulates a ball rolling under the image surface")
    print("   - Good for removing gradual background variations")
    
    rolling_corrected = test_background_method(
        gray_image, "ROLLING_BALL", advanced_processor.rolling_ball_3d, 50
    )
    if rolling_corrected is not None:
        methods.append(("Rolling Ball", rolling_corrected))
    
    # 4. RESTORE method
    print(f"\n{'='*50}")
    print("4. RESTORE METHOD")
    print("   - Identifies dark regions as background automatically")
    print("   - Subtracts average background value")
    
    restore_corrected = test_background_method(
        gray_image, "RESTORE", advanced_processor.restore_method, 5.0
    )
    if restore_corrected is not None:
        methods.append(("RESTORE", restore_corrected))
    
    # 5. Homomorphic Filter
    print(f"\n{'='*50}")
    print("5. HOMOMORPHIC METHOD")
    print("   - Uses frequency domain filtering")
    print("   - Good for multiplicative illumination effects")
    
    homomorphic_corrected = test_background_method(
        gray_image, "HOMOMORPHIC", advanced_processor.homomorphic_filter, 30, 0.5, 2.0
    )
    if homomorphic_corrected is not None:
        methods.append(("Homomorphic", homomorphic_corrected))
    
    # Show how to configure the method
    print(f"\n{'='*50}")
    print("HOW TO SELECT BACKGROUND CORRECTION METHOD:")
    print("=" * 50)
    print("The method is controlled by the 'background_correction_method' setting.")
    print("Available options:")
    for method in BackgroundCorrectionMethod:
        print(f"  - {method.name}: {method}")
    
    print("\nIn YAML configuration file:")
    print("processing:")
    print("  background_correction_method: 'none'      # Use default normalization")
    print("  background_correction_method: 'basic'     # Use BaSiC (NMF)")
    print("  background_correction_method: 'rolling_ball'  # Use rolling ball")
    print("  background_correction_method: 'restore'   # Use RESTORE method")
    print("  background_correction_method: 'homomorphic'  # Use homomorphic filter")
    
    print("\nParameters for each method:")
    print("  basic_correction_n_components: 1          # Number of NMF components")
    print("  rolling_ball_radius: 50                   # Radius for rolling ball")
    print("  restore_percentile: 5.0                   # Percentile for RESTORE")
    print("  homomorphic_cutoff: 30                    # Cutoff frequency")
    print("  homomorphic_g_low: 0.5                    # Low gain")
    print("  homomorphic_g_high: 2.0                   # High gain")
    
    print(f"\n{'='*50}")
    print("RECOMMENDATION:")
    print("- Start with 'none' (default normalization) - works well for most cases")
    print("- Try 'rolling_ball' for gradual illumination changes")
    print("- Try 'basic' for complex illumination patterns")
    print("- 'restore' and 'homomorphic' are more specialized")
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON:")
    print("Method                    | Saturation Rate | Dynamic Range")
    print("-" * 55)
    for name, img in methods:
        if img is not None:
            sat_rate = 100 * np.sum(img >= 240) / img.size
            p2, p98 = np.percentile(img, [2, 98])
            print(f"{name:24} | {sat_rate:6.2f}%       | {p2:3.0f} - {p98:3.0f}")


if __name__ == "__main__":
    main()
