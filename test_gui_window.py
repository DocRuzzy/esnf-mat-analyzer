#!/usr/bin/env python3
"""
Test script that creates the GUI window briefly and then closes it.
"""

import tkinter as tk
import sys

def test_gui_creation():
    """Test that GUI can be created and destroyed without errors."""
    try:
        print("Creating GUI window...")
        
        # Import the main window
        from esnf_mat_analyzer.gui.main_window import MainWindow
        
        # Create the window
        app = MainWindow()
        
        # Schedule the window to close after 1 second
        def close_after_delay():
            print("✓ GUI window created successfully")
            app.destroy()
            
        app.after(1000, close_after_delay)  # Close after 1 second
        
        # Start the main loop (will exit after 1 second)
        app.mainloop()
        
        print("✅ GUI test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_creation()
    sys.exit(0 if success else 1)
