#!/usr/bin/env python3
"""
Phase 1 Performance Validation - Quick Demo

This script runs a quick demonstration of Phase 1 benchmarking with a smaller dataset
to validate the complete benchmarking framework without taking too long.

Author: ESNF Mat Analyzer Team
Date: August 2025
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmark_phase1_performance import PerformanceBenchmark

def run_quick_phase1_demo():
    """Run a quick Phase 1 demo with a small dataset"""
    
    print("=== PHASE 1 PERFORMANCE VALIDATION - QUICK DEMO ===")
    print("This is a scaled-down demonstration of the full Phase 1 validation")
    print("Dataset size: 10 images (instead of 654)")
    print("Focus: Validate framework functionality and generate sample reports\n")
    
    # Setup output directory
    output_dir = Path("benchmark_results") / f"phase1_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize benchmark with custom dataset size
    benchmark = PerformanceBenchmark(output_dir)
    
    try:
        # Override the dataset generation to use smaller number
        print("\n1. Generating small synthetic dataset (10 images)...")
        ground_truth_data = benchmark.dataset_generator.generate_rulers2023_style_dataset(10)
        
        print("2. Benchmarking ESNF Mat Analyzer (Default config only)...")
        # Test only the default configuration to save time
        from esnf_mat_analyzer.config.config_manager import get_default_config
        from esnf_mat_analyzer.main import setup_dependencies
        
        config = get_default_config()
        analyzer = setup_dependencies(config)
        
        esnf_results = []
        for i, gt_data in enumerate(ground_truth_data[:3]):  # Only test 3 images
            print(f"   Processing image {i+1}/3...")
            result = benchmark._process_single_image_esnf(analyzer, gt_data)
            esnf_results.append(result)
        
        print("3. Benchmarking baseline methods...")
        baseline_results = benchmark._benchmark_baseline_methods(ground_truth_data[:3])
        
        print("4. Calculating metrics...")
        # Create results structure
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': 3,
            'methods_compared': ['Default ESNF'] + list(baseline_results.keys()),
            'metrics': {}
        }
        
        # Calculate metrics for ESNF
        esnf_metrics = benchmark._calculate_method_metrics(esnf_results, ground_truth_data[:3], 'Default ESNF')
        results['metrics']['Default ESNF'] = esnf_metrics
        
        # Calculate metrics for baseline methods
        for method_name, method_results in baseline_results.items():
            metrics = benchmark._calculate_method_metrics(method_results, ground_truth_data[:3], method_name)
            results['metrics'][method_name] = metrics
        
        # Calculate relative performance
        results['relative_performance'] = benchmark._calculate_relative_performance(results['metrics'])
        
        print("5. Generating reports...")
        benchmark._generate_performance_reports(results)
        
        print("\n" + "="*60)
        print("PHASE 1 DEMO RESULTS SUMMARY")
        print("="*60)
        
        for method, metrics in results['metrics'].items():
            print(f"\n{method}:")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
            print(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.3f}s")
            if 'mAPE_scale' in metrics:
                print(f"  Scale mAPE: {metrics['mAPE_scale']:.4f}")
        
        # Best performers
        if 'best_performers' in results['relative_performance']:
            print(f"\nBest Performers:")
            for metric, (method, value) in results['relative_performance']['best_performers'].items():
                print(f"  {metric.replace('_', ' ').title()}: {method}")
        
        print(f"\n📁 Full results and reports available in: {output_dir}")
        print(f"📊 Key files:")
        print(f"   - benchmark_summary.md: Human-readable summary")
        print(f"   - benchmark_results.json: Complete results data")
        print(f"   - performance_comparison.png: Visualization plots")
        print(f"   - performance_metrics.csv: Detailed metrics table")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    
    start_time = time.time()
    
    results = run_quick_phase1_demo()
    
    total_time = time.time() - start_time
    
    if results:
        print(f"\n🎉 Phase 1 demo completed successfully in {total_time:.1f} seconds!")
        print(f"✅ Framework validated and ready for full-scale benchmarking")
        print(f"\nTo run the complete Phase 1 validation with 654 images:")
        print(f"   python benchmark_phase1_performance.py")
    else:
        print(f"\n❌ Demo failed after {total_time:.1f} seconds")

if __name__ == "__main__":
    main()
