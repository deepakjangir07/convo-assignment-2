#!/bin/bash

echo "🚀 Financial QA System Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit, torch, transformers" &> /dev/null; then
    echo "⚠️  Some dependencies are missing"
    echo "Installing requirements..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error installing requirements"
        exit 1
    fi
    echo "✅ Requirements installed"
else
    echo "✅ Dependencies found"
fi

echo
echo "🎉 Starting Streamlit app..."
echo
echo "The app will open in your default browser"
echo "Press Ctrl+C to stop the app"
echo

# Start the Streamlit app
streamlit run app.py 