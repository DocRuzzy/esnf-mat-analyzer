#!/usr/bin/env python3
"""
Script to run the ESNF Mat Analyzer GUI application.
This script ensures the proper Python environment is used.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the GUI application
from esnf_mat_analyzer.gui.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
