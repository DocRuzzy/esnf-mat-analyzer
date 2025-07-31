#!/usr/bin/env python3
"""
Simple test to verify GUI functionality without actually opening the GUI.
"""

import sys
import os

def test_gui_imports():
    """Test that all GUI imports work correctly."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import tkinter as tk
        from PIL import Image, ImageTk
        print("✓ GUI libraries imported successfully")
        
        # Test analyzer imports
        from esnf_mat_analyzer.main import create_config, setup_dependencies
        print("✓ Analyzer imports successful")
        
        # Test GUI module import
        from esnf_mat_analyzer.gui.main_window import MainWindow
        print("✓ GUI module imported successfully")
        
        # Test config and analyzer creation
        config = create_config()
        analyzer = setup_dependencies(config)
        print("✓ Analyzer setup successful")
        
        print("\n✅ All tests passed! GUI should work correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_imports()
    sys.exit(0 if success else 1)
