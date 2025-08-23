@echo off
echo 🚀 Financial QA System Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import streamlit, torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some dependencies are missing
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Error installing requirements
        pause
        exit /b 1
    )
    echo ✅ Requirements installed
) else (
    echo ✅ Dependencies found
)

echo.
echo 🎉 Starting Streamlit app...
echo.
echo The app will open in your default browser
echo Press Ctrl+C to stop the app
echo.

REM Start the Streamlit app
streamlit run app.py

pause 