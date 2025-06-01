@echo off
echo Starting Jupyter Lab...
CALL conda activate intel_cnn
IF ERRORLEVEL 1 (
    echo Failed to activate environment.
    pause
    EXIT /B 1
)
jupyter lab
IF ERRORLEVEL 1 (
    echo Failed to start Jupyter Lab.
    pause
    EXIT /B 1
)