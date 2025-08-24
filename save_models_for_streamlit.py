#!/usr/bin/env python3
"""
Standalone Model Saver for ConvoAI Assignment 2
Run this after completing the notebook to save models for Streamlit app.
"""

import os
import pickle
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def save_models_for_streamlit():
    """Save all models and components for use in Streamlit app."""
    print("🚀 Saving Models for Streamlit App")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/rag', exist_ok=True)
    os.makedirs('models/fine_tuned', exist_ok=True)
    
    try:
        # Load components (assuming they exist from notebook execution)
        print("📂 Loading components...")
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Load text chunks
        with open('data/processed_financials.txt', 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = text_splitter.split_text(full_text)
        print(f"✅ Text chunks created: {len(chunks)} chunks")
        
        # Load base model
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        print("✅ Base model loaded")
        
        # Save RAG components
        print("\n💾 Saving RAG components...")
        
        # Save embedding model
        embedding_model.save('models/rag/embedding_model')
        print("✅ Embedding model saved")
        
        # Save text chunks
        with open('models/rag/chunks.pkl', 'wb') as f:
            pickle.dump(chunks, f)
        print("✅ Text chunks saved")
        
        # Save base model
        base_model_path = 'models/rag/base_model'
        os.makedirs(base_model_path, exist_ok=True)
        base_model.save_pretrained(base_model_path)
        tokenizer.save_pretrained(base_model_path)
        print("✅ Base model saved")
        
        # Check for fine-tuned model
        print("\n🔍 Checking for fine-tuned model...")
        
        if os.path.exists('./ft_model_distilgpt2_adapters'):
            print("✅ Fine-tuned model found, saving...")
            
            from peft import PeftModel
            ft_model_merged = PeftModel.from_pretrained(base_model, './ft_model_distilgpt2_adapters')
            ft_model_merged = ft_model_merged.merge_and_unload()
            
            ft_model_merged.save_pretrained('models/fine_tuned/merged_model')
            tokenizer.save_pretrained('models/fine_tuned/merged_model')
            print("✅ Fine-tuned model saved")
        else:
            print("⚠️  Fine-tuned model not found, skipping...")
        
        # Create model configuration
        print("\n📋 Creating model configuration...")
        
        model_config = {
            "rag": {
                "embedding_model": "all-MiniLM-L6-v2",
                "base_model": "distilgpt2",
                "chunks_file": "chunks.pkl",
                "base_model_path": "base_model",
                "embedding_model_path": "embedding_model"
            },
            "fine_tuned": {
                "model_path": "merged_model",
                "base_model": "distilgpt2",
                "training_method": "LoRA"
            },
            "data": {
                "processed_text": "data/processed_financials.txt",
                "qa_dataset": "qa_dataset.csv"
            },
            "advanced_techniques": {
                "rag": "Hybrid Search (Sparse + Dense Retrieval)",
                "fine_tuning": "Adapter-Based Parameter-Efficient Tuning (LoRA)"
            }
        }
        
        with open('models/model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        print("✅ Model configuration saved")
        
        # Create README
        readme_content = """# Models Directory

This directory contains all models and components for the Streamlit app.

## Structure:
- `rag/`: RAG system components
- `fine_tuned/`: Fine-tuned model
- `model_config.json`: Configuration file

## Usage in Streamlit App:
The app will automatically load these models when initialized.
"""
        
        with open('models/README.md', 'w') as f:
            f.write(readme_content)
        print("✅ README created")
        
        print("\n🎉 All models saved successfully for Streamlit app!")
        print("📁 Check the 'models/' directory for all saved components.")
        print("🚀 Your Streamlit app is now ready to use these models!")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        print("\n💡 Make sure you have run the notebook cells first to create the models.")

if __name__ == "__main__":
    save_models_for_streamlit()
