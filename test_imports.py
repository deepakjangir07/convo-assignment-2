#!/usr/bin/env python3
"""
Test script to isolate import issues.
"""

def test_pymupdf():
    """Test PyMuPDF import."""
    print("🧪 Testing PyMuPDF Import...")
    try:
        import fitz
        print("✅ PyMuPDF (fitz) imported successfully")
        print(f"📚 Version: {fitz.version}")
        return True
    except Exception as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False

def test_model_utils():
    """Test model_utils import."""
    print("\n🔧 Testing Model Utils Import...")
    try:
        from model_utils import get_qa_system
        print("✅ model_utils imported successfully")
        print("✅ get_qa_system function available")
        return True
    except Exception as e:
        print(f"❌ model_utils import failed: {e}")
        return False

def test_app_import():
    """Test app.py import."""
    print("\n📱 Testing App Import...")
    try:
        import app
        print("✅ app.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def main():
    """Run all import tests."""
    print("🚀 Import Issue Isolation Test")
    print("=" * 50)
    
    # Test 1: PyMuPDF
    pymupdf_works = test_pymupdf()
    
    # Test 2: Model Utils
    model_utils_works = test_model_utils()
    
    # Test 3: App
    app_works = test_app_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print(f"  • PyMuPDF: {'✅ Working' if pymupdf_works else '❌ Failed'}")
    print(f"  • Model Utils: {'✅ Working' if model_utils_works else '❌ Failed'}")
    print(f"  • App: {'✅ Working' if app_works else '❌ Failed'}")
    
    if pymupdf_works and model_utils_works and app_works:
        print("\n🎉 All imports working! No issues found.")
    else:
        print("\n⚠️  Import issues detected. Check the errors above.")
    
    print("\n💡 Next Steps:")
    if not pymupdf_works:
        print("1. Fix PyMuPDF import issue")
    if not model_utils_works:
        print("2. Fix model_utils import issue")
    if not app_works:
        print("3. Fix app.py import issue")

if __name__ == "__main__":
    main() 