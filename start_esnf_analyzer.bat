@echo off
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
