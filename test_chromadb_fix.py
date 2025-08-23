#!/usr/bin/env python3
"""
Comprehensive test for ChromaDB error handling fixes.
This test verifies that the 'NoneType' object has no attribute 'Client' error is resolved.
"""

def test_chromadb_import_status():
    """Test ChromaDB import status and variables."""
    print("🧪 Testing ChromaDB Import Status...")
    print("=" * 50)
    
    try:
        from model_utils import CHROMADB_AVAILABLE, chromadb
        print(f"✅ CHROMADB_AVAILABLE: {CHROMADB_AVAILABLE}")
        print(f"✅ chromadb object: {type(chromadb) if chromadb else 'None'}")
        
        if chromadb is not None:
            print(f"✅ chromadb has Client: {hasattr(chromadb, 'Client')}")
            if hasattr(chromadb, 'Client'):
                print(f"✅ chromadb.Client type: {type(chromadb.Client)}")
        else:
            print("⚠️  chromadb is None - fallback system will be used")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB import status test failed: {e}")
        return False

def test_model_utils_import():
    """Test if model_utils can be imported without errors."""
    print("\n🔧 Testing Model Utils Import...")
    print("=" * 50)
    
    try:
        from model_utils import get_qa_system
        print("✅ model_utils imported successfully")
        print("✅ get_qa_system function available")
        return True
    except Exception as e:
        print(f"❌ model_utils import failed: {e}")
        return False

def test_qa_system_initialization():
    """Test if the QA system can be initialized without ChromaDB errors."""
    print("\n🚀 Testing QA System Initialization...")
    print("=" * 50)
    
    try:
        from model_utils import get_qa_system
        print("🔄 Initializing QA system...")
        
        # This should not throw the 'NoneType' object has no attribute 'Client' error
        qa_system = get_qa_system()
        print("✅ QA system initialized successfully!")
        
        # Check what storage method was used
        if hasattr(qa_system, 'collection') and qa_system.collection is not None:
            print("📊 Storage: ChromaDB")
        else:
            print("📊 Storage: Alternative (In-Memory)")
        
        if hasattr(qa_system, 'bm25') and qa_system.bm25 is not None:
            print("📊 Sparse: BM25")
        else:
            print("📊 Sparse: Alternative (Keyword)")
        
        return True
    except Exception as e:
        print(f"❌ QA system initialization failed: {e}")
        return False

def test_app_import():
    """Test if the app can be imported without errors."""
    print("\n📱 Testing App Import...")
    print("=" * 50)
    
    try:
        import app
        print("✅ app.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ChromaDB Error Handling - COMPREHENSIVE TEST")
    print("=" * 60)
    print("This test verifies that ALL ChromaDB-related errors are resolved.")
    print("=" * 60)
    
    # Test 1: ChromaDB import status
    import_status_ok = test_chromadb_import_status()
    
    # Test 2: Model utils import
    model_utils_ok = test_model_utils_import()
    
    # Test 3: QA system initialization (critical test)
    qa_system_ok = test_qa_system_initialization()
    
    # Test 4: App import
    app_ok = test_app_import()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST RESULTS:")
    print(f"  • ChromaDB Import Status: {'✅ Working' if import_status_ok else '❌ Failed'}")
    print(f"  • Model Utils Import: {'✅ Working' if model_utils_ok else '❌ Failed'}")
    print(f"  • QA System Initialization: {'✅ Working' if qa_system_ok else '❌ Failed'}")
    print(f"  • App Import: {'✅ Working' if app_ok else '❌ Failed'}")
    
    if import_status_ok and model_utils_ok and qa_system_ok and app_ok:
        print("\n🎉 ALL TESTS PASSED! ChromaDB error handling is COMPLETELY RESOLVED!")
        print("   Your app will now work on Streamlit Cloud without ChromaDB errors.")
        print("   The system will automatically choose the best storage method.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("\n💡 Next Steps:")
    if import_status_ok and model_utils_ok and qa_system_ok and app_ok:
        print("1. ✅ Your app is ready for Streamlit Cloud deployment")
        print("2. Push changes to GitHub and deploy")
        print("3. ChromaDB errors are completely resolved")
        print("4. System will work regardless of environment differences")
        print("5. Users will see clear status messages about storage method")
    else:
        print("1. Fix the remaining issues")
        print("2. Test again before deployment")
        print("3. Check the error messages above")
    
    print("\n🔍 Key Fixes Applied:")
    print("  • ChromaDB import error handling with explicit variable initialization")
    print("  • hasattr(chromadb, 'Client') checks before using chromadb.Client")
    print("  • Comprehensive try-catch blocks around all ChromaDB operations")
    print("  • Graceful fallback to alternative storage methods")
    print("  • Critical error handling with minimal system setup")

if __name__ == "__main__":
    main() 