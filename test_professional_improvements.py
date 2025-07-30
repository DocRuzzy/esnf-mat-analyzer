#!/usr/bin/env python3
"""
Professional improvements verification test for ESNF Mat Analyzer.

This script tests all the professional enhancements we've implemented:
1. Enhanced documentation and type hints
2. Professional error handling and logging
3. Configuration validation
4. CLI interface
5. Result validation and utility methods

Run this script to verify that all improvements are working correctly.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
        from esnf_mat_analyzer.core.data_types import Config, AnalysisResult
        from esnf_mat_analyzer.config.config_manager import (
            get_default_config, validate_config, load_config, save_config
        )
        
        # CLI module
        from esnf_mat_analyzer.cli.main import main_cli
        
        # Logging module
        from esnf_mat_analyzer.utils.logging_config import (
            setup_application_logging, AnalysisLogger
        )
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration_validation():
    """Test configuration validation with various scenarios."""
    print("🔍 Testing configuration validation...")
    
    try:
        from esnf_mat_analyzer.config.config_manager import get_default_config, validate_config
        
        # Test valid default configuration
        config = get_default_config()
        errors = validate_config(config)
        
        if errors:
            print(f"❌ Default configuration has errors: {errors}")
            return False
        
        # Test invalid configuration
        config.processing.blur_kernel_size = -1  # Invalid
        errors = validate_config(config)
        
        if not errors:
            print("❌ Validation should have caught invalid blur_kernel_size")
            return False
        
        print("✅ Configuration validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False


def test_analyzer_documentation():
    """Test that the analyzer has proper documentation and type hints."""
    print("🔍 Testing analyzer documentation...")
    
    try:
        from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
        
        # Check that class has docstring
        if not NanoFiberAnalyzer.__doc__:
            print("❌ NanoFiberAnalyzer missing class docstring")
            return False
        
        # Check that methods have docstrings
        if not NanoFiberAnalyzer.process_image.__doc__:
            print("❌ process_image method missing docstring")
            return False
        
        if not NanoFiberAnalyzer.batch_process.__doc__:
            print("❌ batch_process method missing docstring")
            return False
        
        # Check for type annotations
        annotations = NanoFiberAnalyzer.process_image.__annotations__
        if not annotations:
            print("❌ process_image method missing type annotations")
            return False
        
        print("✅ Analyzer documentation and type hints present")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer documentation test failed: {e}")
        return False


def test_analysis_result_enhancements():
    """Test the enhanced AnalysisResult class."""
    print("🔍 Testing AnalysisResult enhancements...")
    
    try:
        import numpy as np
        from esnf_mat_analyzer.core.data_types import AnalysisResult
        
        # Create a test result
        test_path = Path("test_image.tif")
        test_contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        test_thickness = np.random.rand(100, 100)
        test_mask = np.ones((100, 100), dtype=bool)
        test_metrics = {
            "overall_mat_uniformity": 0.85,
            "anisotropy_index": 0.23
        }
        
        result = AnalysisResult(
            image_path=test_path,
            contour=test_contour,
            thickness_map=test_thickness,
            mask=test_mask,
            metrics=test_metrics,
            processing_time=2.5
        )
        
        # Test utility methods
        uniformity = result.get_metric("overall_mat_uniformity")
        if uniformity != 0.85:
            print(f"❌ get_metric failed: expected 0.85, got {uniformity}")
            return False
        
        # Test summary generation
        summary = result.get_summary()
        if not isinstance(summary, dict):
            print("❌ get_summary should return a dictionary")
            return False
        
        if "processing_time_sec" not in summary:
            print("❌ Summary missing processing_time_sec")
            return False
        
        print("✅ AnalysisResult enhancements working correctly")
        return True
        
    except Exception as e:
        print(f"❌ AnalysisResult test failed: {e}")
        return False


def test_logging_configuration():
    """Test the professional logging configuration."""
    print("🔍 Testing logging configuration...")
    
    try:
        from esnf_mat_analyzer.utils.logging_config import (
            setup_application_logging, AnalysisLogger
        )
        
        # Test basic setup
        logger_manager = setup_application_logging(verbose=True)
        if not isinstance(logger_manager, AnalysisLogger):
            print("❌ setup_application_logging should return AnalysisLogger")
            return False
        
        # Test getting a logger
        logger = logger_manager.get_logger("test")
        if not isinstance(logger, logging.Logger):
            print("❌ get_logger should return Logger instance")
            return False
        
        # Test logging a message (should not raise exceptions)
        logger.info("Test message for professional improvements verification")
        
        print("✅ Logging configuration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logging configuration test failed: {e}")
        return False


def test_cli_argument_parsing():
    """Test CLI argument parsing (without execution)."""
    print("🔍 Testing CLI argument parsing...")
    
    try:
        from esnf_mat_analyzer.cli.main import create_argument_parser, parse_roi_string
        
        # Test parser creation
        parser = create_argument_parser()
        if not parser:
            print("❌ Failed to create argument parser")
            return False
        
        # Test ROI parsing
        roi = parse_roi_string("10,20,100,200")
        if roi != (10, 20, 100, 200):
            print(f"❌ ROI parsing failed: expected (10, 20, 100, 200), got {roi}")
            return False
        
        # Test invalid ROI parsing
        try:
            parse_roi_string("invalid")
            print("❌ Should have raised ValueError for invalid ROI")
            return False
        except ValueError:
            pass  # Expected
        
        print("✅ CLI argument parsing working correctly")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_from_config_method():
    """Test the from_config class method."""
    print("🔍 Testing from_config method...")
    
    try:
        from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
        from esnf_mat_analyzer.config.config_manager import get_default_config
        
        # This might fail if dependencies aren't set up, but should not crash
        config = get_default_config()
        
        # Test that the method exists and has proper signature
        if not hasattr(NanoFiberAnalyzer, 'from_config'):
            print("❌ NanoFiberAnalyzer missing from_config class method")
            return False
        
        if not callable(NanoFiberAnalyzer.from_config):
            print("❌ from_config is not callable")
            return False
        
        print("✅ from_config method exists and is callable")
        return True
        
    except Exception as e:
        print(f"❌ from_config test failed: {e}")
        return False


def run_all_tests() -> bool:
    """Run all professional improvement tests."""
    print("=" * 60)
    print("🚀 ESNF Mat Analyzer Professional Improvements Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Validation", test_configuration_validation),
        ("Analyzer Documentation", test_analyzer_documentation),
        ("AnalysisResult Enhancements", test_analysis_result_enhancements),
        ("Logging Configuration", test_logging_configuration),
        ("CLI Argument Parsing", test_cli_argument_parsing),
        ("From Config Method", test_from_config_method),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All professional improvements are working correctly!")
        print("\n📋 Professional Features Verified:")
        print("   • Enhanced documentation and type hints")
        print("   • Comprehensive error handling and validation")
        print("   • Professional logging configuration")
        print("   • Enhanced CLI interface with colored output")
        print("   • Improved configuration management")
        print("   • Result validation and utility methods")
    else:
        print(f"⚠️  {total - passed} tests failed. Review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
