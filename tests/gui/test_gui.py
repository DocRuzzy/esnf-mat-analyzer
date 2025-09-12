#!/usr/bin/env python3
"""
Test script to launch the GUI application.
"""

if __name__ == "__main__":
    try:
        from esnf_mat_analyzer.gui.main_window import MainWindow
        print("Starting GUI application...")
        app = MainWindow()
        app.mainloop()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the package is installed: pip install -e .")
    except Exception as e:
        print(f"Error starting GUI: {e}")
