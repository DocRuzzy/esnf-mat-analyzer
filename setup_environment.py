#!/usr/bin/env python3
"""
Setup script for ESNF Mat Analyzer development environment.

This script ensures the package is properly installed in development mode
and all dependencies are available. Run this whenever you:
- Clone the repository
- Switch branches 
- Set up a new development environment
- Encounter import errors

For JOSS compliance, this script provides:
- Reproducible environment setup
- Dependency verification
- Platform compatibility checks
- Clear error reporting

Author: ESNF Mat Analyzer Team
License: MIT (as specified in LICENSE file)
"""

import subprocess
import sys
import platform
import os
from pathlib import Path
import importlib.util
import pkg_resources
from typing import List, Dict, Optional, Tuple

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class EnvironmentSetup:
    """Main setup class for ESNF Mat Analyzer development environment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.platform_info = f"{platform.system()} {platform.release()}"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def print_header(self):
        """Print a formatted header for the setup process."""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("  ESNF Mat Analyzer - Development Environment Setup")
        print("=" * 70)
        print(f"{Colors.END}")
        print(f"Python Version: {Colors.GREEN}{self.python_version}{Colors.END}")
        print(f"Platform: {Colors.GREEN}{self.platform_info}{Colors.END}")
        print(f"Project Root: {Colors.GREEN}{self.project_root}{Colors.END}")
        print()

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        print(f"{Colors.BLUE}🔍 Checking Python version...{Colors.END}")
        
        min_version = (3, 8)
        current_version = (sys.version_info.major, sys.version_info.minor)
        
        if current_version >= min_version:
            print(f"  {Colors.GREEN}✅ Python {self.python_version} meets requirements (>= 3.8){Colors.END}")
            return True
        else:
            error_msg = f"Python {self.python_version} is too old. Requires Python >= 3.8"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False

    def check_git_repository(self) -> bool:
        """Check if we're in a valid git repository."""
        print(f"{Colors.BLUE}🔍 Checking git repository...{Colors.END}")
        
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--git-dir"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Get current branch
                branch_result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_branch = branch_result.stdout.strip()
                print(f"  {Colors.GREEN}✅ Valid git repository (branch: {current_branch}){Colors.END}")
                return True
            except subprocess.CalledProcessError:
                warning_msg = "Git repository may be corrupted"
                self.warnings.append(warning_msg)
                print(f"  {Colors.YELLOW}⚠️  {warning_msg}{Colors.END}")
                return False
        else:
            warning_msg = "Not in a git repository"
            self.warnings.append(warning_msg)
            print(f"  {Colors.YELLOW}⚠️  {warning_msg}{Colors.END}")
            return False

    def check_project_structure(self) -> bool:
        """Verify essential project files exist."""
        print(f"{Colors.BLUE}🔍 Checking project structure...{Colors.END}")
        
        required_files = [
            "pyproject.toml",
            "requirements.txt", 
            "esnf_mat_analyzer/__init__.py",
            "esnf_mat_analyzer/main.py",
            "esnf_mat_analyzer/gui/main_window.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False
        else:
            print(f"  {Colors.GREEN}✅ All required project files found{Colors.END}")
            return True

    def install_package_editable(self) -> bool:
        """Install the package in editable mode."""
        print(f"{Colors.BLUE}📦 Installing package in editable mode...{Colors.END}")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  {Colors.GREEN}✅ Package installed successfully{Colors.END}")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install package: {e.stderr}"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False

    def install_requirements(self) -> bool:
        """Install package requirements."""
        print(f"{Colors.BLUE}📋 Installing requirements...{Colors.END}")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            warning_msg = "requirements.txt not found, skipping"
            self.warnings.append(warning_msg)
            print(f"  {Colors.YELLOW}⚠️  {warning_msg}{Colors.END}")
            return True
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  {Colors.GREEN}✅ Requirements installed successfully{Colors.END}")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install requirements: {e.stderr}"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False

    def verify_imports(self) -> bool:
        """Verify that key modules can be imported."""
        print(f"{Colors.BLUE}🔍 Verifying package imports...{Colors.END}")
        
        key_modules = [
            "esnf_mat_analyzer",
            "esnf_mat_analyzer.main",
            "esnf_mat_analyzer.gui.main_window",
            "esnf_mat_analyzer.core.analyzer",
            "esnf_mat_analyzer.config.config_manager"
        ]
        
        failed_imports = []
        for module_name in key_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    failed_imports.append(module_name)
                    continue
                    
                # Try to actually import it
                importlib.import_module(module_name)
                
            except ImportError as e:
                failed_imports.append(f"{module_name} ({e})")
        
        if failed_imports:
            error_msg = f"Failed to import modules: {', '.join(failed_imports)}"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False
        else:
            print(f"  {Colors.GREEN}✅ All key modules import successfully{Colors.END}")
            return True

    def check_optional_dependencies(self) -> bool:
        """Check for optional dependencies that enhance functionality."""
        print(f"{Colors.BLUE}🔍 Checking optional dependencies...{Colors.END}")
        
        optional_deps = {
            "matplotlib": "Required for visualization and plotting",
            "opencv-python": "Required for advanced image processing", 
            "scikit-image": "Required for image analysis algorithms",
            "numpy": "Required for numerical computations",
            "scipy": "Required for scientific computing",
            "tkinter": "Required for GUI (usually included with Python)"
        }
        
        missing_optional = []
        for dep, description in optional_deps.items():
            try:
                if dep == "tkinter":
                    import tkinter
                else:
                    importlib.import_module(dep.replace("-", "_"))
                print(f"  {Colors.GREEN}✅ {dep}{Colors.END}")
            except ImportError:
                missing_optional.append(f"{dep}: {description}")
                print(f"  {Colors.YELLOW}⚠️  {dep} not available{Colors.END}")
        
        if missing_optional:
            warning_msg = f"Optional dependencies missing: {len(missing_optional)} items"
            self.warnings.append(warning_msg)
            print(f"  {Colors.YELLOW}⚠️  Some optional features may not work{Colors.END}")
        
        return True

    def create_run_script(self) -> bool:
        """Create or update the run script with proper path handling."""
        print(f"{Colors.BLUE}📝 Creating run script...{Colors.END}")
        
        run_script_content = '''#!/usr/bin/env python3
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
'''
        
        try:
            run_script_path = self.project_root / "run_esnf_analyzer.py"
            with open(run_script_path, 'w', encoding='utf-8') as f:
                f.write(run_script_content)
            
            # Make executable on Unix systems
            if platform.system() != "Windows":
                os.chmod(run_script_path, 0o755)
            
            print(f"  {Colors.GREEN}✅ Run script created: run_esnf_analyzer.py{Colors.END}")
            return True
        except Exception as e:
            error_msg = f"Failed to create run script: {e}"
            self.errors.append(error_msg)
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            return False

    def create_batch_scripts(self) -> bool:
        """Create platform-specific convenience scripts."""
        print(f"{Colors.BLUE}📝 Creating convenience scripts...{Colors.END}")
        
        # Windows batch file
        if platform.system() == "Windows":
            batch_content = '''@echo off
echo Setting up ESNF Mat Analyzer environment...
python setup_environment.py
if %ERRORLEVEL% equ 0 (
    echo.
    echo Environment setup complete!
    echo Starting application...
    python run_esnf_analyzer.py
) else (
    echo.
    echo Environment setup failed. Please check the error messages above.
    pause
)
'''
            try:
                batch_path = self.project_root / "start_esnf_analyzer.bat"
                with open(batch_path, 'w', encoding='utf-8') as f:
                    f.write(batch_content)
                print(f"  {Colors.GREEN}✅ Windows batch script created{Colors.END}")
            except Exception as e:
                warning_msg = f"Failed to create batch script: {e}"
                self.warnings.append(warning_msg)
                print(f"  {Colors.YELLOW}⚠️  {warning_msg}{Colors.END}")
        
        # Unix shell script
        else:
            shell_content = '''#!/bin/bash
echo "Setting up ESNF Mat Analyzer environment..."
python3 setup_environment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Environment setup complete!"
    echo "Starting application..."
    python3 run_esnf_analyzer.py
else
    echo ""
    echo "Environment setup failed. Please check the error messages above."
    exit 1
fi
'''
            try:
                shell_path = self.project_root / "start_esnf_analyzer.sh"
                with open(shell_path, 'w', encoding='utf-8') as f:
                    f.write(shell_content)
                os.chmod(shell_path, 0o755)
                print(f"  {Colors.GREEN}✅ Unix shell script created{Colors.END}")
            except Exception as e:
                warning_msg = f"Failed to create shell script: {e}"
                self.warnings.append(warning_msg)
                print(f"  {Colors.YELLOW}⚠️  {warning_msg}{Colors.END}")
        
        return True

    def print_summary(self) -> bool:
        """Print a summary of the setup process."""
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("  Setup Summary")
        print("=" * 70)
        print(f"{Colors.END}")
        
        if not self.errors:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 Environment setup completed successfully!{Colors.END}")
            print()
            print("You can now:")
            print(f"  • Run the GUI: {Colors.CYAN}python run_esnf_analyzer.py{Colors.END}")
            if platform.system() == "Windows":
                print(f"  • Use shortcut: {Colors.CYAN}start_esnf_analyzer.bat{Colors.END}")
            else:
                print(f"  • Use shortcut: {Colors.CYAN}./start_esnf_analyzer.sh{Colors.END}")
            print(f"  • Import in Python: {Colors.CYAN}from esnf_mat_analyzer.gui.main_window import MainWindow{Colors.END}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.END}")
                for warning in self.warnings:
                    print(f"  ⚠️  {warning}")
            
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Environment setup failed!{Colors.END}")
            print(f"\n{Colors.RED}Errors ({len(self.errors)}):{Colors.END}")
            for error in self.errors:
                print(f"  ❌ {error}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.END}")
                for warning in self.warnings:
                    print(f"  ⚠️  {warning}")
            
            print(f"\n{Colors.BLUE}Recommended actions:{Colors.END}")
            print("  1. Check Python version (>= 3.8 required)")
            print("  2. Ensure you're in the correct project directory")
            print("  3. Check internet connection for package downloads")
            print("  4. Try running with administrator/sudo privileges")
            
            return False

def main():
    """Main setup function."""
    setup = EnvironmentSetup()
    
    # Print header
    setup.print_header()
    
    # Run all setup steps
    steps = [
        setup.check_python_version,
        setup.check_git_repository,
        setup.check_project_structure,
        setup.install_package_editable,
        setup.install_requirements,
        setup.verify_imports,
        setup.check_optional_dependencies,
        setup.create_run_script,
        setup.create_batch_scripts
    ]
    
    success = True
    for step in steps:
        if not step():
            success = False
            # Continue with other steps even if one fails
    
    # Print summary
    final_success = setup.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if final_success else 1)

if __name__ == "__main__":
    main()
