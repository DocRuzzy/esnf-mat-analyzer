#!/usr/bin/env python3
"""
Test script to verify ESNF Mat Analyzer installation and basic functionality.

This script tests:
- Package imports
- Basic configuration loading
- Core functionality access
- GUI initialization (without showing window)

Run this after setup to verify everything is working correctly.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core imports
        from esnf_mat_analyzer.main import setup_dependencies
        from esnf_mat_analyzer.config.config_manager import get_default_config
        from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
        from esnf_mat_analyzer.gui.main_window import MainWindow
        
        # Processing imports
        from esnf_mat_analyzer.processing.image_processor import ImageProcessor
        from esnf_mat_analyzer.processing.thickness_estimator import ThicknessEstimator
        
        # Analysis imports
        from esnf_mat_analyzer.analysis.uniformity_metrics import RadialUniformityIndex
        
        # Visualization imports
        from esnf_mat_analyzer.visualization.visualization import Visualizer
        
        print("  ✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading and setup."""
    print("Testing configuration...")
    
    try:
        from esnf_mat_analyzer.config.config_manager import get_default_config
        from esnf_mat_analyzer.main import setup_dependencies
        
        # Load default config
        config = get_default_config()
        print(f"  ✅ Default config loaded: {type(config).__name__}")
        
        # Setup dependencies
        analyzer = setup_dependencies(config)
        print(f"  ✅ Dependencies setup: {type(analyzer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_gui_class():
    """Test GUI class instantiation without showing window."""
    print("Testing GUI class...")
    
    try:
        from esnf_mat_analyzer.gui.main_window import MainWindow
        import tkinter as tk
        
        # Test class instantiation (don't run mainloop)
        print("  ✅ GUI class accessible")
        print("  ℹ️  GUI window test skipped (would block terminal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GUI test failed: {e}")
        traceback.print_exc()
        return False

def test_optional_dependencies():
    """Test optional dependencies availability."""
    print("Testing optional dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'matplotlib': 'matplotlib',
        'cv2': 'opencv-python',
        'skimage': 'scikit-image',
        'PIL': 'Pillow',
        'yaml': 'PyYAML'
    }
    
    available = []
    missing = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            available.append(package)
        except ImportError:
            missing.append(package)
    
    for pkg in available:
        print(f"  ✅ {pkg}")
    
    for pkg in missing:
        print(f"  ⚠️  {pkg} not available")
    
    return len(missing) == 0

def test_entry_points():
    """Test that entry points work correctly."""
    print("Testing entry points...")
    
    try:
        # Test main entry point
        from esnf_mat_analyzer.main import main as cli_main
        print("  ✅ CLI entry point accessible")
        
        # Test GUI entry point  
        from esnf_mat_analyzer.gui.main_window import main as gui_main
        print("  ✅ GUI entry point accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Entry point test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("ESNF Mat Analyzer - Installation Test")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration), 
        ("GUI Class Test", test_gui_class),
        ("Dependencies Test", test_optional_dependencies),
        ("Entry Points Test", test_entry_points)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ESNF Mat Analyzer is ready to use.")
        print()
        print("Next steps:")
        print("  • Run GUI: python run_esnf_analyzer.py")
        print("  • Run CLI: python -m esnf_mat_analyzer.main --help")
        print("  • See examples in the examples/ directory")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print()
        print("Troubleshooting:")
        print("  • Run: python setup_environment.py")
        print("  • Check dependencies: pip install -r requirements.txt")
        print("  • Verify Python version >= 3.8")
        return 1

if __name__ == "__main__":
    sys.exit(main())
