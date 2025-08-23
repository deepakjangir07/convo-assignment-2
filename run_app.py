#!/usr/bin/env python3
"""
Launcher script for the Financial QA System Streamlit app.
This script handles common setup issues and provides helpful error messages.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'sentence_transformers',
        'chromadb',
        'langchain'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.util.find_spec(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_data_directory():
    """Check if data directory exists and has content."""
    data_dir = "data"
    data_file = os.path.join(data_dir, "processed_financials.txt")
    
    if not os.path.exists(data_dir):
        print(f"⚠️  Data directory '{data_dir}' not found.")
        print("Creating empty data directory...")
        os.makedirs(data_dir, exist_ok=True)
        return False
    
    if not os.path.exists(data_file):
        print(f"⚠️  Data file '{data_file}' not found.")
        print("The system will work but may not have financial data to query.")
        return False
    
    print(f"✅ Data directory and file found")
    return True

def check_models():
    """Check if fine-tuned models are available."""
    model_path = "./ft_model_distilgpt2_adapters"
    
    if os.path.exists(model_path):
        print(f"✅ Fine-tuned model found at {model_path}")
        return True
    else:
        print(f"⚠️  Fine-tuned model not found at {model_path}")
        print("The system will use the base model only.")
        return False

def install_requirements():
    """Install requirements if needed."""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_streamlit():
    """Run the Streamlit app."""
    print("\n🚀 Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    """Main launcher function."""
    print("🚀 Financial QA System Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\nWould you like to install missing dependencies? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            if not install_requirements():
                sys.exit(1)
        else:
            print("Please install dependencies manually and try again.")
            sys.exit(1)
    
    # Check data
    print("\n📁 Checking data...")
    check_data_directory()
    
    # Check models
    print("\n🤖 Checking models...")
    check_models()
    
    print("\n" + "=" * 40)
    print("🎉 All checks passed! Starting the app...")
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main() 