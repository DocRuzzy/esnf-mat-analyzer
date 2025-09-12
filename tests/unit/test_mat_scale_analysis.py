#!/usr/bin/env python3
"""
Test script for mat-scale uniformity analysis features.

This script tests the new FFT anisotropy, GLCM texture, and power spectral density
analysis features for fiber mat uniformity assessment.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create synthetic test data for mat-scale analysis."""
    logger.info("Creating synthetic test data...")
    
    # Create different types of fiber mat patterns
    size = 256
    
    # 1. Uniform (isotropic) mat
    uniform_mat = np.random.normal(100, 10, (size, size))
    uniform_mat = np.clip(uniform_mat, 0, 255).astype(np.uint8)
    
    # 2. Anisotropic mat (preferential fiber direction)
    x, y = np.meshgrid(np.linspace(0, 10*np.pi, size), np.linspace(0, 10*np.pi, size))
    anisotropic_mat = 100 + 20 * np.sin(x) + np.random.normal(0, 5, (size, size))
    anisotropic_mat = np.clip(anisotropic_mat, 0, 255).astype(np.uint8)
    
    # 3. Textured mat (high contrast variations)
    textured_mat = np.zeros((size, size))
    for i in range(0, size, 20):
        for j in range(0, size, 20):
            block_val = np.random.randint(50, 150)
            textured_mat[i:i+20, j:j+20] = block_val
    textured_mat += np.random.normal(0, 10, (size, size))
    textured_mat = np.clip(textured_mat, 0, 255).astype(np.uint8)
    
    # Create masks (full image for all)
    mask = np.ones((size, size), dtype=np.uint8) * 255
    
    return {
        'uniform': (uniform_mat, mask),
        'anisotropic': (anisotropic_mat, mask),
        'textured': (textured_mat, mask)
    }

def test_mat_anisotropy():
    """Test the mat anisotropy analyzer."""
    logger.info("Testing Mat Anisotropy Analyzer...")
    
    try:
        from esnf_mat_analyzer.analysis.mat_anisotropy import MatAnisotropyAnalyzer
        
        analyzer = MatAnisotropyAnalyzer()
        test_data = create_test_data()
        
        for name, (thickness_map, mask) in test_data.items():
            logger.info(f"Analyzing {name} mat...")
            results = analyzer.analyze_mat_anisotropy(thickness_map, mask)
            
            print(f"\n{name.upper()} MAT - Anisotropy Results:")
            print("-" * 40)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error testing mat anisotropy: {e}")
        import traceback
        traceback.print_exc()

def test_mat_texture():
    """Test the mat texture analyzer."""
    logger.info("Testing Mat Texture Analyzer...")
    
    try:
        from esnf_mat_analyzer.analysis.mat_anisotropy import MatTextureAnalyzer
        
        analyzer = MatTextureAnalyzer()
        test_data = create_test_data()
        
        for name, (thickness_map, mask) in test_data.items():
            logger.info(f"Analyzing {name} mat texture...")
            results = analyzer.analyze_mat_texture(thickness_map, mask)
            
            print(f"\n{name.upper()} MAT - Texture Results:")
            print("-" * 40)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error testing mat texture: {e}")
        import traceback
        traceback.print_exc()

def test_power_spectral_density():
    """Test the power spectral density analyzer."""
    logger.info("Testing Power Spectral Density Analyzer...")
    
    try:
        from esnf_mat_analyzer.analysis.mat_anisotropy import PowerSpectralDensityAnalyzer
        
        analyzer = PowerSpectralDensityAnalyzer()
        test_data = create_test_data()
        
        for name, (thickness_map, mask) in test_data.items():
            logger.info(f"Analyzing {name} mat PSD...")
            results = analyzer.analyze_psd(thickness_map, mask)
            
            print(f"\n{name.upper()} MAT - PSD Results:")
            print("-" * 40)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error testing PSD: {e}")
        import traceback
        traceback.print_exc()

def test_integrated_mat_analysis():
    """Test the integrated mat uniformity analyzer."""
    logger.info("Testing Integrated Mat Uniformity Analyzer...")
    
    try:
        from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
        from esnf_mat_analyzer.core.data_types import UniformityConfig
        
        # Create config
        analyzer = MultiScaleUniformityAnalyzer()
        test_data = create_test_data()
        
        for name, (thickness_map, mask) in test_data.items():
            logger.info(f"Performing comprehensive analysis on {name} mat...")
            results = analyzer.analyze_multiscale_uniformity(thickness_map, mask)
            
            print(f"\n{name.upper()} MAT - Comprehensive Analysis:")
            print("=" * 50)
            
            # Overall uniformity score
            if 'overall_mat_uniformity' in results:
                score = results['overall_mat_uniformity']
                rating = get_uniformity_rating(score)
                print(f"Overall Uniformity Score: {score:.4f} ({rating})")
                print()
            
            # Group results by category
            anisotropy_results = {k: v for k, v in results.items() if k.startswith('anisotropy_')}
            texture_results = {k: v for k, v in results.items() if k.startswith('texture_')}
            psd_results = {k: v for k, v in results.items() if k.startswith('psd_')}
            
            if anisotropy_results:
                print("Anisotropy Analysis:")
                for key, value in anisotropy_results.items():
                    clean_name = key.replace('anisotropy_', '').replace('_', ' ').title()
                    if isinstance(value, float):
                        print(f"  {clean_name}: {value:.4f}")
                    else:
                        print(f"  {clean_name}: {value}")
                print()
            
            if texture_results:
                print("Texture Analysis:")
                for key, value in texture_results.items():
                    clean_name = key.replace('texture_', '').replace('_', ' ').title()
                    if isinstance(value, float):
                        print(f"  {clean_name}: {value:.4f}")
                    else:
                        print(f"  {clean_name}: {value}")
                print()
            
            if psd_results:
                print("Power Spectral Density Analysis:")
                for key, value in psd_results.items():
                    clean_name = key.replace('psd_', '').replace('_', ' ').title()
                    if isinstance(value, float):
                        print(f"  {clean_name}: {value:.4f}")
                    else:
                        print(f"  {clean_name}: {value}")
                print()
    
    except Exception as e:
        logger.error(f"Error testing integrated mat analysis: {e}")
        import traceback
        traceback.print_exc()

def get_uniformity_rating(score):
    """Convert uniformity score to a descriptive rating."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    elif score >= 0.2:
        return "Poor"
    else:
        return "Very Poor"

def test_with_real_image():
    """Test mat-scale analysis with a real sample image."""
    logger.info("Testing with real sample image...")
    
    # Look for sample images
    sample_paths = [
        Path("tests/Samples/square.png"),
        Path("tests/Samples/bracket.png"),
        Path("tests/Samples/U10.7G5W4.png")
    ]
    
    for sample_path in sample_paths:
        if sample_path.exists():
            logger.info(f"Testing with {sample_path}")
            
            try:
                # Load image
                image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logger.warning(f"Could not load {sample_path}")
                    continue
                
                # Create mask (simple threshold)
                _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Test integrated analysis
                from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
                
                analyzer = MultiScaleUniformityAnalyzer()
                results = analyzer.analyze_multiscale_uniformity(image, mask)
                
                print(f"\nREAL IMAGE ({sample_path.name}) - Mat Analysis:")
                print("=" * 50)
                
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                break  # Test with first available image
                
            except Exception as e:
                logger.error(f"Error testing with real image {sample_path}: {e}")
    
    else:
        logger.warning("No sample images found for testing")

def visualize_test_data():
    """Create visualizations of the test data."""
    logger.info("Creating visualizations...")
    
    try:
        test_data = create_test_data()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, (thickness_map, mask)) in enumerate(test_data.items()):
            im = axes[i].imshow(thickness_map, cmap='viridis', aspect='equal')
            axes[i].set_title(f'{name.title()} Mat')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('mat_scale_test_data.png', dpi=150, bbox_inches='tight')
        logger.info("Test data visualization saved as 'mat_scale_test_data.png'")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def main():
    """Run all mat-scale analysis tests."""
    print("Mat-Scale Uniformity Analysis Test Suite")
    print("=" * 50)
    
    # Create visualizations
    visualize_test_data()
    
    # Run individual component tests
    test_mat_anisotropy()
    test_mat_texture()
    test_power_spectral_density()
    
    # Run integrated analysis test
    test_integrated_mat_analysis()
    
    # Test with real image if available
    test_with_real_image()
    
    print("\n" + "=" * 50)
    print("Mat-scale analysis testing completed!")
    print("Check the output above for analysis results.")

if __name__ == "__main__":
    main()
