"""
Configuration file for the Financial QA System.
Centralize all system settings here for easy modification.
"""

import os

# Model Configuration
MODEL_CONFIG = {
    "base_model_name": "distilgpt2",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "fine_tuned_model_path": "./ft_model_distilgpt2_adapters",
    "max_new_tokens": 100,
    "temperature": 0.1,
    "top_k": 10,
    "do_sample": True,
    "num_return_sequences": 1
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 50,
    "top_k_retrieval": 5,
    "rrf_k": 60,  # Reciprocal Rank Fusion parameter
    "collection_name": "financials_rag"
}

# Text Processing Configuration
TEXT_CONFIG = {
    "max_length": 1024,
    "truncation": True,
    "remove_patterns": [
        r'Page \d+ of \d+',
        r'Microsoft Corporation\s+Form 10-K',
        r'\s*\n\s*'
    ]
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Financial QA System",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_chat_history": 5,
    "max_context_display": 500
}

# File Paths
PATHS = {
    "data_directory": "data",
    "processed_data_file": "data/processed_financials.txt",
    "evaluation_results_file": "evaluation_results.csv",
    "chunks_export_file": "data_chunks.csv",
    "statistics_export_file": "data_statistics.csv"
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "device_map": "auto",
    "torch_dtype": "float32",
    "batch_size": 1,
    "gradient_accumulation_steps": 2
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "sample_questions": [
        "What was Microsoft's revenue in 2023?",
        "What are the primary strategic risks related to AI development?",
        "How much did Microsoft spend on research and development in 2023?",
        "What is the capital of France?",
        "Compare the net income of 2023 and 2022."
    ],
    "metrics": ["confidence", "response_time", "answer_quality"]
}

# Advanced Options
ADVANCED_CONFIG = {
    "enable_hybrid_retrieval": True,
    "enable_rrf_fusion": True,
    "enable_context_truncation": True,
    "enable_confidence_scoring": True,
    "enable_response_time_tracking": True
}

# Environment Variables
def get_env_config():
    """Get configuration from environment variables."""
    return {
        "wandb_disabled": os.getenv("WANDB_DISABLED", "true"),
        "transformers_cache_dir": os.getenv("TRANSFORMERS_CACHE_DIR", None),
        "torch_cache_dir": os.getenv("TORCH_HOME", None),
        "hf_home": os.getenv("HF_HOME", None)
    }

# Validation
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check required paths
    for path_name, path_value in PATHS.items():
        if "directory" in path_name:
            if not os.path.exists(path_value):
                os.makedirs(path_value, exist_ok=True)
    
    # Check model paths
    if not os.path.exists(MODEL_CONFIG["fine_tuned_model_path"]):
        print(f"⚠️  Fine-tuned model not found at {MODEL_CONFIG['fine_tuned_model_path']}")
    
    # Validate numeric values
    if MODEL_CONFIG["temperature"] < 0 or MODEL_CONFIG["temperature"] > 1:
        errors.append("Temperature must be between 0 and 1")
    
    if RETRIEVAL_CONFIG["chunk_size"] <= 0:
        errors.append("Chunk size must be positive")
    
    if RETRIEVAL_CONFIG["chunk_overlap"] >= RETRIEVAL_CONFIG["chunk_size"]:
        errors.append("Chunk overlap must be less than chunk size")
    
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True

# Export all configurations
__all__ = [
    'MODEL_CONFIG',
    'RETRIEVAL_CONFIG', 
    'TEXT_CONFIG',
    'UI_CONFIG',
    'PATHS',
    'PERFORMANCE_CONFIG',
    'EVALUATION_CONFIG',
    'ADVANCED_CONFIG',
    'get_env_config',
    'validate_config'
] 