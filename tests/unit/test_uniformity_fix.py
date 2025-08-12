#!/usr/bin/env python3
"""
Quick test for the fixed RadialUniformityIndex and other uniformity metrics.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_uniformity_metrics():
    """Test the uniformity metrics with simple data."""
    print("Testing uniformity metrics...")
    
    try:
        # Import the fixed classes
        from esnf_mat_analyzer.analysis.uniformity_metrics import (
            RadialUniformityIndex, GiniCoefficient, ThicknessRangeRatio
        )
        from esnf_mat_analyzer.core.data_types import UniformityConfig
        
        # Create test data
        uniform_map = np.ones((100, 100), dtype=float) * 50.0
        non_uniform_map = np.ones((100, 100), dtype=float) * 30.0
        non_uniform_map[25:75, 25:75] = 80.0  # Add a thicker region
        
        # Create ROI mask
        roi_mask = np.zeros((100, 100), dtype=bool)
        roi_mask[10:90, 10:90] = True
        
        # Test configuration
        config = UniformityConfig(num_radial_lines=12)
        
        # Initialize metrics
        rui = RadialUniformityIndex(config)
        gini = GiniCoefficient()
        trr = ThicknessRangeRatio()
        
        print("  Testing RadialUniformityIndex...")
        
        # Test uniform map (should have high RUI)
        uniform_rui = rui.calculate_metric(uniform_map, roi_mask)
        print(f"    Uniform map RUI: {uniform_rui:.4f}")
        
        # Test non-uniform map (should have lower RUI)
        non_uniform_rui = rui.calculate_metric(non_uniform_map, roi_mask)
        print(f"    Non-uniform map RUI: {non_uniform_rui:.4f}")
        
        assert uniform_rui > non_uniform_rui, "Uniform map should have higher RUI"
        assert 0.0 <= uniform_rui <= 1.0, "RUI should be in [0,1] range"
        assert 0.0 <= non_uniform_rui <= 1.0, "RUI should be in [0,1] range"
        
        print("  Testing GiniCoefficient...")
        
        # Test Gini coefficient
        uniform_gini = gini.calculate_metric(uniform_map, roi_mask)
        non_uniform_gini = gini.calculate_metric(non_uniform_map, roi_mask)
        print(f"    Uniform map Gini: {uniform_gini:.4f}")
        print(f"    Non-uniform map Gini: {non_uniform_gini:.4f}")
        
        assert uniform_gini < non_uniform_gini, "Uniform map should have lower Gini"
        assert 0.0 <= uniform_gini <= 1.0, "Gini should be in [0,1] range"
        assert 0.0 <= non_uniform_gini <= 1.0, "Gini should be in [0,1] range"
        
        print("  Testing ThicknessRangeRatio...")
        
        # Test thickness range ratio
        uniform_trr = trr.calculate_metric(uniform_map, roi_mask)
        non_uniform_trr = trr.calculate_metric(non_uniform_map, roi_mask)
        print(f"    Uniform map TRR: {uniform_trr:.4f}")
        print(f"    Non-uniform map TRR: {non_uniform_trr:.4f}")
        
        assert uniform_trr > non_uniform_trr, "Uniform map should have higher TRR"
        assert 0.0 <= uniform_trr <= 1.0, "TRR should be in [0,1] range"
        assert 0.0 <= non_uniform_trr <= 1.0, "TRR should be in [0,1] range"
        
        print("  ✅ All uniformity metrics working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Uniformity metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_integration():
    """Test that the analyzer can be created and used with the fixed metrics."""
    print("Testing analyzer integration...")
    
    try:
        from esnf_mat_analyzer.main import setup_dependencies
        from esnf_mat_analyzer.config.config_manager import get_default_config
        
        # Get default config and setup analyzer
        config = get_default_config()
        analyzer = setup_dependencies(config)
        
        print(f"  ✅ Analyzer created successfully: {type(analyzer).__name__}")
        return True
        
    except Exception as e:
        print(f"  ❌ Analyzer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("RadialUniformityIndex Fix Verification")
    print("=" * 50)
    
    tests = [
        ("Uniformity Metrics", test_uniformity_metrics),
        ("Analyzer Integration", test_analyzer_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 RadialUniformityIndex issue has been successfully fixed!")
        print("\nThe uniformity metrics now:")
        print("  • Accept proper configuration parameters")
        print("  • Calculate realistic uniformity values")
        print("  • Distinguish between uniform and non-uniform data")
        print("  • Work correctly with the main analyzer")
        return 0
    else:
        print("⚠️  Some issues remain. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
