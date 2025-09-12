#!/usr/bin/env python3
"""
Entry point for ESNF Mat Analyzer GUI application.

This script automatically handles Python path setup and provides
a robust entry point for the application.

For JOSS compliance, this script:
- Handles import path issues gracefully
- Provides clear error messages
- Auto-recovers from common setup problems
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Add project root to Python path if needed."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

def auto_install_if_needed(project_root):
    """Automatically install package if import fails."""
    try:
        from esnf_mat_analyzer.gui.main_window import MainWindow
        return MainWindow
    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
        print("Attempting to install package in development mode...")
        
        import subprocess
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], cwd=project_root, check=True)
            print("Package installed successfully!")
            
            # Try import again
            from esnf_mat_analyzer.gui.main_window import MainWindow
            return MainWindow
            
        except subprocess.CalledProcessError as install_error:
            print(f"Failed to install package: {install_error}")
            print("Please run 'python setup_environment.py' to fix the environment.")
            sys.exit(1)
        except ImportError as import_error:
            print(f"Import still failing after installation: {import_error}")
            print("Please check your Python environment and dependencies.")
            sys.exit(1)

def main():
    """Main entry point for the application."""
    print("Starting ESNF Mat Analyzer...")
    
    # Setup paths
    project_root = setup_python_path()
    
    # Get the main window class (with auto-install if needed)
    MainWindow = auto_install_if_needed(project_root)
    
    # Create and run the application
    try:
        app = MainWindow()
        print("GUI started successfully!")
        app.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
