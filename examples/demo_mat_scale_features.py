#!/usr/bin/env python3
"""
Demo script showcasing the new mat-scale uniformity analysis features.

This script demonstrates the enhanced ESNF mat analyzer with:
- FFT-based anisotropy analysis
- GLCM texture analysis
- Power spectral density analysis
- Composite uniformity scoring
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_mat_scale_analysis():
    """Demonstrate the new mat-scale analysis capabilities."""
    print("ESNF Mat Analyzer - Mat-Scale Uniformity Analysis Demo")
    print("=" * 60)
    
    try:
        from esnf_mat_analyzer.main import setup_dependencies
        from esnf_mat_analyzer.core.data_types import Config
        
        # Initialize the analyzer
        config = Config()
        analyzer = setup_dependencies(config)
        
        print("✓ Analyzer initialized with mat-scale analysis capabilities")
        
        # Test with available sample images
        sample_paths = [
            Path("tests/Samples/square.png"),
            Path("tests/Samples/bracket.png"),
            Path("tests/Samples/U10.7G5W4.png")
        ]
        
        for sample_path in sample_paths:
            if sample_path.exists():
                print(f"\n📊 Analyzing: {sample_path.name}")
                print("-" * 40)
                
                # Process the image
                result = analyzer.process_image(sample_path)
                
                # Extract mat-scale metrics
                traditional_metrics = {k: v for k, v in result.metrics.items() 
                                     if not k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity'))}
                
                mat_metrics = {k: v for k, v in result.metrics.items() 
                              if k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity'))}
                
                print(f"Traditional metrics: {len(traditional_metrics)}")
                print(f"Mat-scale metrics: {len(mat_metrics)}")
                
                # Show key results
                if 'overall_mat_uniformity' in result.metrics:
                    score = result.metrics['overall_mat_uniformity']
                    rating = get_uniformity_rating(score)
                    print(f"\n🎯 Overall Mat Uniformity: {score:.4f} ({rating})")
                
                # Show detailed mat-scale analysis
                print("\n📈 Mat-Scale Analysis Results:")
                
                # Anisotropy
                anisotropy_metrics = {k: v for k, v in result.metrics.items() if k.startswith('anisotropy_')}
                if anisotropy_metrics:
                    print("  FFT Anisotropy:")
                    for key, value in anisotropy_metrics.items():
                        clean_name = key.replace('anisotropy_', '').replace('_', ' ').title()
                        if isinstance(value, float):
                            print(f"    {clean_name}: {value:.4f}")
                
                # Texture
                texture_metrics = {k: v for k, v in result.metrics.items() if k.startswith('texture_')}
                if texture_metrics:
                    print("  GLCM Texture:")
                    key_texture_metrics = ['texture_texture_homogeneity', 'texture_texture_energy', 
                                         'texture_texture_contrast', 'texture_texture_correlation']
                    for key in key_texture_metrics:
                        if key in texture_metrics:
                            clean_name = key.replace('texture_texture_', '').title()
                            print(f"    {clean_name}: {texture_metrics[key]:.4f}")
                
                # PSD
                psd_metrics = {k: v for k, v in result.metrics.items() if k.startswith('psd_')}
                if psd_metrics:
                    print("  Power Spectral Density:")
                    for key, value in psd_metrics.items():
                        clean_name = key.replace('psd_', '').replace('_', ' ').title()
                        if isinstance(value, float):
                            print(f"    {clean_name}: {value:.4f}")
                
                print(f"\n⏱️ Processing time: {result.processing_time:.2f} seconds")
                
                # Break after first successful analysis for demo
                break
        
        else:
            print("⚠️ No sample images found. Please ensure test samples are available.")
            return
        
        print("\n" + "=" * 60)
        print("🎉 Mat-scale analysis demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• FFT-based anisotropy analysis for fiber orientation assessment")
        print("• GLCM texture analysis for surface uniformity characterization")
        print("• Power spectral density analysis for frequency domain characteristics")
        print("• Composite uniformity scoring combining multiple metrics")
        print("• Publication-ready metrics suitable for peer review")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
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

def demonstrate_analysis_types():
    """Show the difference between fiber-scale and mat-scale analysis."""
    print("\n🔍 Analysis Scale Comparison")
    print("=" * 40)
    print("FIBER-SCALE Analysis (Individual Fibers):")
    print("• Fiber diameter measurements")
    print("• Individual fiber morphology")
    print("• Single fiber properties")
    print("• Microscopic features")
    
    print("\nMAT-SCALE Analysis (Whole Mat Structure):")
    print("• Overall mat uniformity")
    print("• Fiber orientation distribution")
    print("• Texture homogeneity")
    print("• Frequency domain characteristics")
    print("• Suitable for manufacturing quality control")

if __name__ == "__main__":
    demonstrate_analysis_types()
    demo_mat_scale_analysis()
