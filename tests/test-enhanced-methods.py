#!/usr/bin/env python3
"""
Test script for comparing enhanced background correction methods
for ESNF mat analysis.

This script loads an image, applies all correction methods,
and creates a comparison visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Tuple

# Import the enhanced processor (adjust path as needed)
# from esnf_mat_analyzer.processing.enhanced_background import EnhancedBackgroundProcessor

def create_test_image() -> np.ndarray:
    """Create a synthetic test image with non-uniform background and fibers."""
    # Create base image
    height, width = 512, 512
    image = np.zeros((height, width), dtype=np.float32)
    
    # Add non-uniform illumination (gradient + gaussian)
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Linear gradient
    gradient = 50 + 100 * (X + Y) / 2
    
    # Gaussian illumination pattern
    center_x, center_y = width // 3, height // 3
    sigma = width / 4
    gaussian = 50 * np.exp(-((X * width - center_x)**2 + (Y * height - center_y)**2) / (2 * sigma**2))
    
    # Combine illumination patterns
    background = gradient + gaussian
    
    # Add synthetic fibers
    np.random.seed(42)
    n_fibers = 50
    for _ in range(n_fibers):
        # Random fiber parameters
        x_start = np.random.randint(0, width)
        y_start = np.random.randint(0, height)
        length = np.random.randint(50, 200)
        angle = np.random.uniform(0, 2 * np.pi)
        intensity = np.random.uniform(150, 255)
        thickness = np.random.randint(2, 5)
        
        # Draw fiber
        x_end = int(x_start + length * np.cos(angle))
        y_end = int(y_start + length * np.sin(angle))
        cv2.line(image, (x_start, y_start), (x_end, y_end), 
                intensity, thickness=thickness)
    
    # Add background to image
    image = image + background
    
    # Add noise
    noise = np.random.normal(0, 5, image.shape)
    image = image + noise
    
    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

def analyze_correction_quality(original: np.ndarray, 
                             corrected: np.ndarray,
                             mask: np.ndarray = None) -> Dict[str, float]:
    """
    Analyze the quality of background correction.
    
    Returns:
        Dictionary with quality metrics
    """
    # If no mask provided, create a simple one based on intensity
    if mask is None:
        _, mask = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask > 0
    
    # Calculate metrics
    metrics = {}
    
    # Dynamic range
    metrics['dynamic_range'] = np.ptp(corrected)
    
    # Contrast (RMS contrast)
    mean_intensity = np.mean(corrected)
    metrics['rms_contrast'] = np.sqrt(np.mean((corrected - mean_intensity)**2)) / mean_intensity
    
    # Background uniformity (CV of background regions)
    if np.any(~mask):
        bg_values = corrected[~mask]
        metrics['bg_uniformity'] = np.std(bg_values) / (np.mean(bg_values) + 1e-6)
    else:
        metrics['bg_uniformity'] = np.inf
    
    # Signal preservation (mean intensity in mask regions)
    if np.any(mask):
        metrics['signal_mean'] = np.mean(corrected[mask])
        metrics['signal_std'] = np.std(corrected[mask])
    else:
        metrics['signal_mean'] = 0
        metrics['signal_std'] = 0
    
    # Edge preservation (gradient magnitude)
    grad_x = cv2.Sobel(corrected, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(corrected, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    metrics['edge_strength'] = np.mean(grad_mag)
    
    # Saturation percentage
    metrics['saturation_pct'] = 100 * np.sum(corrected >= 254) / corrected.size
    
    return metrics

def test_all_methods(image: np.ndarray, 
                    processor) -> Dict[str, Tuple[np.ndarray, Dict, float]]:
    """
    Test all background correction methods.
    
    Returns:
        Dictionary with method names as keys and tuples of 
        (corrected_image, metrics, processing_time) as values
    """
    results = {}
    
    # Test each method
    methods = {
        'Original': lambda img: img,
        'Dual-Scale Morphological': lambda img: processor.dual_scale_morphological_correction(img),
        'Signal-Preserving BaSiC': lambda img: processor.signal_preserving_basic(img),
        'Adaptive Local': lambda img: processor.adaptive_local_background(img),
        'Frequency Selective': lambda img: processor.frequency_selective_filtering(img),
        'Multi-Scale Retinex': lambda img: processor.multi_scale_retinex_mat_aware(img),
        'Hybrid (Auto)': lambda img: processor.hybrid_illumination_correction(img)
    }
    
    for name, method in methods.items():
        print(f"Testing {name}...")
        start_time = time.time()
        
        try:
            corrected = method(image)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = analyze_correction_quality(image, corrected)
            
            results[name] = (corrected, metrics, processing_time)
            print(f"  ✓ Completed in {processing_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results[name] = (image, {}, 0)
    
    return results

def create_comparison_figure(results: Dict[str, Tuple[np.ndarray, Dict, float]], 
                           save_path: str = None):
    """Create a comparison figure showing all methods."""
    n_methods = len(results)
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    for idx, (method_name, (image, metrics, proc_time)) in enumerate(results.items()):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        # Display image
        im = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"{method_name}\n"
                    f"Time: {proc_time:.3f}s\n"
                    f"BG Uniformity: {metrics.get('bg_uniformity', 0):.3f}\n"
                    f"Signal Mean: {metrics.get('signal_mean', 0):.1f}",
                    fontsize=10)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.show()

def create_metrics_comparison_table(results: Dict[str, Tuple[np.ndarray, Dict, float]]):
    """Create a comparison table of metrics."""
    print("\n" + "="*80)
    print("METRICS COMPARISON TABLE")
    print("="*80)
    
    # Header
    print(f"{'Method':<25} {'Time(s)':<8} {'BG Unif.':<10} {'Signal':<10} "
          f"{'Contrast':<10} {'Edges':<10} {'Sat.%':<8}")
    print("-"*80)
    
    # Rows
    for method_name, (_, metrics, proc_time) in results.items():
        if metrics:  # Skip if metrics are empty (error case)
            print(f"{method_name:<25} "
                  f"{proc_time:<8.3f} "
                  f"{metrics.get('bg_uniformity', 0):<10.3f} "
                  f"{metrics.get('signal_mean', 0):<10.1f} "
                  f"{metrics.get('rms_contrast', 0):<10.3f} "
                  f"{metrics.get('edge_strength', 0):<10.1f} "
                  f"{metrics.get('saturation_pct', 0):<8.2f}")
    
    print("="*80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    # Find best method for each metric
    best_uniformity = min(results.items(), 
                         key=lambda x: x[1][1].get('bg_uniformity', float('inf')))[0]
    best_signal = max(results.items(), 
                     key=lambda x: x[1][1].get('signal_mean', 0))[0]
    fastest = min(results.items(), 
                 key=lambda x: x[1][2] if x[0] != 'Original' else float('inf'))[0]
    
    print(f"- Best background uniformity: {best_uniformity}")
    print(f"- Best signal preservation: {best_signal}")
    print(f"- Fastest processing: {fastest}")

def main():
    """Main test function."""
    print("Enhanced Background Correction Methods Test")
    print("==========================================\n")
    
    # Initialize processor
    processor = EnhancedBackgroundProcessor()
    
    # Option 1: Use synthetic test image
    print("Creating synthetic test image...")
    test_image = create_test_image()
    
    # Option 2: Load your own image
    # image_path = "path/to/your/esnf_image.png"
    # test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Test all methods
    results = test_all_methods(test_image, processor)
    
    # Create visualizations
    create_comparison_figure(results, save_path="background_correction_comparison.png")
    create_metrics_comparison_table(results)
    
    # Interactive selection
    print("\nBased on the results above, which method would you like to use?")
    print("Enter the method name or press Enter to use the recommended method.")
    
    # Additional analysis for specific image regions
    print("\nFor more detailed analysis, you can:")
    print("1. Load your actual ESNF mat images")
    print("2. Define ROI masks for targeted correction")
    print("3. Fine-tune parameters for your specific application")
    print("4. Implement the selected method in your GUI")

if __name__ == "__main__":
    main()
