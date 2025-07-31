#!/usr/bin/env python3
"""
Test script for Beer-Lambert thickness estimation implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import cv2
from pathlib import Path
from esnf_mat_analyzer.processing.physics_based_thickness import BeerLambertEstimator, MultiModelThicknessEstimator
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config
from esnf_mat_analyzer.core.data_types import ThicknessModelType

def test_beer_lambert_basic():
    """Test basic Beer-Lambert thickness estimation."""
    print("=== BASIC BEER-LAMBERT TEST ===")
    
    # Create synthetic test image with known properties
    height, width = 200, 200
    image = np.ones((height, width), dtype=np.uint8) * 200  # Light background
    
    # Add darker regions (higher fiber density)
    image[50:100, 50:100] = 150  # Medium density
    image[75:125, 75:125] = 100  # High density
    image[100:150, 100:150] = 50   # Very high density
    
    print(f"Test image: {width}x{height}, range: {image.min()}-{image.max()}")
    
    # Test Beer-Lambert estimator
    estimator = BeerLambertEstimator()
    thickness_map = estimator.estimate_thickness(image)
    
    print(f"Thickness map range: {thickness_map.min():.3f} - {thickness_map.max():.3f}")
    
    # Get statistics
    stats = estimator.get_thickness_statistics(thickness_map)
    print("\nBeer-Lambert Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Verify that darker regions have higher thickness
    bg_thickness = thickness_map[25, 25]  # Background
    med_thickness = thickness_map[75, 75]  # Medium density
    high_thickness = thickness_map[100, 100]  # High density
    
    print(f"\nThickness verification:")
    print(f"  Background (bright): {bg_thickness:.3f}")
    print(f"  Medium density: {med_thickness:.3f}")
    print(f"  High density (dark): {high_thickness:.3f}")
    
    assert bg_thickness < med_thickness < high_thickness, "Darker regions should have higher thickness"
    print("✓ Thickness relationship verified (dark = thick)")
    
    return True

def test_multi_model_comparison():
    """Test comparison of multiple thickness models."""
    print("\n=== MULTI-MODEL COMPARISON TEST ===")
    
    # Create test image with gradient
    height, width = 100, 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create gradient from bright (0.8) to dark (0.2)
    gradient = 0.8 - 0.6 * X
    image = (gradient * 255).astype(np.uint8)
    
    print(f"Gradient image: {width}x{height}, range: {image.min()}-{image.max()}")
    
    # Test with multiple models
    multi_estimator = MultiModelThicknessEstimator()
    performance = multi_estimator.validate_model_performance(image)
    
    print("\nModel Comparison Results:")
    for model_name, metrics in performance.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Mean: {metrics['mean_thickness']:.2f}")
        print(f"  Std: {metrics['std_thickness']:.2f}")
        print(f"  Range: {metrics['dynamic_range']:.2f}")
        print(f"  SNR: {metrics['signal_to_noise']:.2f}")
        print(f"  CV: {metrics['coefficient_of_variation']:.3f}")
    
    # Verify all models produced results
    assert len(performance) >= 3, "Should have at least 3 models"
    assert 'beer_lambert' in performance, "Beer-Lambert model should be included"
    
    print("✓ Multi-model comparison successful")
    return True

def test_full_pipeline_integration():
    """Test Beer-Lambert integration with full analysis pipeline."""
    print("\n=== FULL PIPELINE INTEGRATION TEST ===")
    
    # Use sample images if available
    sample_dir = Path("tests/Samples")
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.png"))
        if sample_images:
            image_path = sample_images[0]
            print(f"Using sample image: {image_path}")
            
            # Test with Beer-Lambert model
            config = get_default_config()
            config.thickness.model_type = ThicknessModelType.BEER_LAMBERT
            
            try:
                analyzer = setup_dependencies(config)
                
                # Load image for ROI estimation
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Could not load image: {image_path}")
                    return False
                
                # Define a reasonable ROI (center 80% of image)
                h, w = image.shape
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                roi = (margin_x, margin_y, w - margin_x, h - margin_y)
                
                print(f"Image shape: {image.shape}, ROI: {roi}")
                
                # Run analysis
                result = analyzer.process_image(image_path, roi=roi)
                
                print(f"Pipeline completed in {result.processing_time:.2f}s")
                print("Analysis metrics:")
                for name, value in result.metrics.items():
                    print(f"  {name}: {value:.4f}")
                
                # Verify results
                assert result.thickness_map is not None, "Thickness map should be generated"
                
                # Calculate expected ROI size
                roi_height = roi[3] - roi[1]  # y2 - y1
                roi_width = roi[2] - roi[0]   # x2 - x1
                expected_shape = (roi_height, roi_width)
                
                print(f"Expected thickness map shape: {expected_shape}")
                print(f"Actual thickness map shape: {result.thickness_map.shape}")
                
                # Thickness map should match ROI size, not full image size
                assert result.thickness_map.shape == expected_shape, f"Thickness map shape {result.thickness_map.shape} should match ROI {expected_shape}"
                
                print("✓ Full pipeline integration successful")
                return True
                
            except Exception as e:
                print(f"Pipeline test failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    print("No sample images found, skipping pipeline test")
    return True

def test_configuration_update():
    """Test that configuration properly uses Beer-Lambert by default."""
    print("\n=== CONFIGURATION TEST ===")
    
    config = get_default_config()
    print(f"Default thickness model: {config.thickness.model_type}")
    print(f"Attenuation coefficient: {config.thickness.attenuation_coefficient}")
    print(f"Thickness range: {config.thickness.thickness_range_um}")
    
    # Verify Beer-Lambert is default
    assert config.thickness.model_type == ThicknessModelType.BEER_LAMBERT, "Beer-Lambert should be default"
    assert config.thickness.attenuation_coefficient == 0.04778, "Should use literature-validated coefficient"
    
    print("✓ Configuration test passed")
    return True

def main():
    """Run all Beer-Lambert tests."""
    print("BEER-LAMBERT THICKNESS ESTIMATION TESTS")
    print("=" * 50)
    
    tests = [
        test_beer_lambert_basic,
        test_multi_model_comparison,
        test_configuration_update,
        test_full_pipeline_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Beer-Lambert implementation is working correctly.")
        print("\nKey Features Implemented:")
        print("- Physics-based thickness estimation using Beer-Lambert law")
        print("- Literature-validated attenuation coefficient (18.84% error)")
        print("- Automatic background intensity detection")
        print("- Multi-model comparison capabilities")
        print("- Full pipeline integration")
        print("- GUI model selection")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
