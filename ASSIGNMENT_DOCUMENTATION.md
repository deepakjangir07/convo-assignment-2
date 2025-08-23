# 📊 Assignment 2: Comparative Financial QA System - RAG vs Fine-Tuning
## Group 46: Hybrid Search + Adapter-Based Parameter-Efficient Tuning

### 🎯 **Assignment Overview**
This project implements and compares two advanced approaches for financial question answering:
1. **Retrieval-Augmented Generation (RAG)** with Hybrid Search (Sparse + Dense Retrieval)
2. **Fine-tuned Language Model** using Adapter-Based Parameter-Efficient Tuning

Both systems are built using open-source technologies and evaluated on Microsoft's financial statements from 2022-2023.

---

## 🏗️ **System Architecture**

### **1. Data Collection & Preprocessing**
- **Source**: Microsoft 10-K reports (2022-2023)
- **Format**: PDF documents converted to structured text
- **Processing**: 
  - OCR text extraction using PyMuPDF
  - Noise removal (headers, footers, page numbers)
  - Text segmentation into logical sections
  - Chunking with configurable sizes (400 tokens with 50 token overlap)

### **2. RAG System Implementation**

#### **2.1 Data Processing**
- **Chunk Sizes**: 400 tokens (primary), 100 tokens (alternative)
- **Metadata**: Unique IDs, source documents, chunk positions
- **Text Cleaning**: Regex-based noise removal and normalization

#### **2.2 Embedding & Indexing**
- **Dense Vector Store**: ChromaDB with sentence-transformers
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Sparse Index**: BM25 algorithm for keyword retrieval
- **Hybrid Approach**: Combines both retrieval methods

#### **2.3 Advanced RAG Technique: Hybrid Search (Sparse + Dense Retrieval)**

**Implementation Details:**
```python
def hybrid_retrieval(self, query, top_k=5):
    """Performs hybrid retrieval using BM25 and vector search, fused with RRF."""
    
    # Dense Retrieval (Vector Similarity)
    query_embedding = self.embedding_model.encode(processed_query).tolist()
    dense_results = self.collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )
    
    # Sparse Retrieval (BM25)
    tokenized_query = processed_query.split()
    bm25_scores = self.bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    # RRF Fusion (Reciprocal Rank Fusion)
    fused_scores = {}
    k = 60  # RRF parameter
    for i, doc in enumerate(dense_docs):
        fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (k + i + 1)
    for i, doc in enumerate(sparse_docs):
        fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (k + i + 1)
    
    return reranked_results
```

**Advantages:**
- **Balanced Recall & Precision**: BM25 catches keyword-specific queries, vectors handle semantic similarity
- **Robust Performance**: Less sensitive to embedding quality variations
- **Domain Adaptability**: Works well with financial terminology and numerical data

#### **2.4 Response Generation**
- **Model**: DistilGPT-2 (82M parameters)
- **Prompt Engineering**: Context + Question + Answer format
- **Token Management**: 1024 token context window with truncation
- **Generation Parameters**: Temperature 0.1, Top-k 10

#### **2.5 Guardrail Implementation**
- **Output-side Guardrail**: Confidence scoring based on answer length and content
- **Implementation**: Flags non-committal or very short answers
- **Threshold**: 0.9 for confident answers, 0.4 for low-confidence

### **3. Fine-tuned Model System Implementation**

#### **3.1 Q/A Dataset Preparation**
- **50 Q&A Pairs**: Comprehensive coverage of financial metrics
- **Categories**: Revenue, costs, profitability, balance sheet, cash flow, segments, risks
- **Confidence Levels**: High (35), Medium (10), Low (5)
- **Format**: Instruction-following style for fine-tuning

#### **3.2 Model Selection**
- **RAG Model**: DistilGPT-2 (open-source, lightweight)
- **Parameters**: 82M (suitable for fine-tuning on consumer hardware)
- **Architecture**: Transformer-based with causal language modeling

#### **3.3 Advanced Fine-tuning Technique: Adapter-Based Parameter-Efficient Tuning**

**Implementation Details:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # Rank of LoRA matrices
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.1,       # Dropout for regularization
    target_modules=["c_attn"],  # Target attention layers
    bias="none",            # Don't train bias terms
    task_type="CAUSAL_LM",  # Causal language modeling
)

ft_model = get_peft_model(base_model, lora_config)
```

**Training Configuration:**
- **Learning Rate**: 5e-5
- **Batch Size**: 4 (with gradient accumulation)
- **Epochs**: 10
- **Optimization**: AdamW with weight decay
- **Mixed Precision**: FP16 for efficiency

**Advantages:**
- **Parameter Efficiency**: Only 0.18% of parameters trained (147K vs 82M)
- **Memory Efficient**: Minimal GPU memory requirements
- **Fast Training**: Reduced training time and computational cost
- **Model Preservation**: RAG model capabilities maintained

#### **3.4 Guardrail Implementation**
- **Input-side Guardrail**: Query validation and filtering
- **Implementation**: Checks for financial domain relevance
- **Output-side Guardrail**: Confidence scoring and hallucination detection

---

## 🧪 **Testing & Evaluation**

### **4.1 Mandatory Test Questions**

#### **M1: Relevant, High-Confidence**
- **Question**: "What was Microsoft's revenue in 2023?"
- **Expected**: Clear factual answer with high confidence
- **Purpose**: Test basic factual retrieval and generation

#### **M2: Relevant, Low-Confidence**
- **Question**: "What are the primary strategic risks related to AI development?"
- **Expected**: Ambiguous information requiring interpretation
- **Purpose**: Test handling of uncertain or sparse information

#### **M3: Irrelevant**
- **Question**: "What is the capital of France?"
- **Expected**: Recognition of out-of-scope queries
- **Purpose**: Test robustness and domain boundary awareness

### **4.2 Extended Evaluation**
- **Total Questions**: 15+ questions across all categories
- **Metrics**: Accuracy, confidence, response time, correctness
- **Categories**: Revenue, costs, profitability, risks, comparisons, irrelevant

### **4.3 Results Table Format**
| Question | Method | Answer | Confidence | Time (s) | Correct (Y/N) |
|----------|--------|--------|------------|----------|---------------|
| Revenue in 2023? | RAG | $211.9B | 0.92 | 0.50 | Y |
| Revenue in 2023? | Fine-tune | $211.9B | 0.93 | 0.41 | Y |
| AI risks? | RAG | [Detailed risks] | 0.81 | 0.79 | Y |
| Capital of France? | RAG | Not in scope | 0.35 | 0.46 | Y |

---

## 📊 **Performance Analysis**

### **RAG System Strengths**
- **Factual Grounding**: Direct access to source documents
- **Adaptability**: Easy to update with new financial data
- **Transparency**: Retrieval context visible to users
- **Robustness**: Handles out-of-domain queries gracefully

### **Fine-tuned Model Strengths**
- **Speed**: Faster inference (no retrieval step)
- **Fluency**: More natural, coherent responses
- **Consistency**: Stable performance across similar questions
- **Efficiency**: Lower computational overhead during inference

### **Trade-offs Analysis**
- **Accuracy vs Speed**: RAG more accurate, Fine-tuned faster
- **Memory vs Performance**: RAG requires more memory, Fine-tuned more efficient
- **Maintenance vs Quality**: RAG easier to update, Fine-tuned requires retraining
- **Domain vs Generalization**: RAG domain-specific, Fine-tuned more generalizable

---

## 🔧 **Technical Implementation Details**

### **Dependencies & Libraries**
```python
# Core ML
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2

# RAG Components
langchain==0.0.350
chromadb==0.4.15
rank_bm25==0.2.2

# Fine-tuning
peft==0.6.0
trl==0.7.4
accelerate==0.24.1

# UI & Processing
streamlit==1.28.1
PyMuPDF==1.23.8
pandas==2.1.3
```

### **File Structure**
```
financial-qa-system/
├── app.py                      # Main Streamlit application
├── model_utils.py              # Core QA system
├── qa_dataset_creator.py       # Dataset generation
├── assignment_evaluation.py    # Evaluation framework
├── config.py                   # Configuration management
├── requirements.txt            # Dependencies
├── README.md                   # User documentation
├── ASSIGNMENT_DOCUMENTATION.md # This file
├── data/                       # Financial documents
├── ft_model_distilgpt2_adapters/  # Fine-tuned model
└── ConvoAI_Assignment2.ipynb  # Original notebook
```

---

## 🚀 **Usage Instructions**

### **1. Setup & Installation**
```bash
# Clone repository
git clone <repository-url>
cd financial-qa-system

# Install dependencies
pip install -r requirements.txt

# Create dataset
python qa_dataset_creator.py

# Run evaluation
python assignment_evaluation.py

# Launch Streamlit app
streamlit run app.py
```

### **2. Running Evaluations**
```bash
# Full assignment evaluation
python assignment_evaluation.py

# Individual components
python -c "from assignment_evaluation import AssignmentEvaluator; eval = AssignmentEvaluator(); eval.run_mandatory_tests()"
```

### **3. Web Interface**
- **URL**: http://localhost:8501
- **Features**: Interactive QA, model comparison, evaluation dashboard
- **Navigation**: Home, Ask Questions, Evaluation, Data Management, Settings

---

## 📈 **Evaluation Results & Insights**

### **Key Findings**
1. **Hybrid Retrieval**: Combines best of both worlds (BM25 + Dense)
2. **LoRA Fine-tuning**: Efficient parameter updates with minimal resource usage
3. **Domain Adaptation**: Both systems show strong performance on financial queries
4. **Robustness**: Effective handling of irrelevant and ambiguous questions

### **Performance Metrics**
- **RAG System**: Higher accuracy, longer response times
- **Fine-tuned Model**: Faster inference, consistent quality
- **Hybrid Retrieval**: Balanced performance across query types
- **Guardrails**: Effective filtering of inappropriate content

---

## 🔮 **Future Enhancements**

### **Short-term Improvements**
- Enhanced correctness evaluation using semantic similarity
- Dynamic chunk size adaptation based on query complexity
- Advanced prompt engineering for better answer generation

### **Long-term Vision**
- Multi-language support for international financial documents
- Real-time data integration from financial APIs
- Advanced reasoning capabilities for complex financial analysis
- Collaborative annotation and feedback systems

---

## 📚 **References & Resources**

### **Technical Papers**
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "Reciprocal Rank Fusion" (Cormack et al., 2009)

### **Open-source Models**
- DistilGPT-2: https://huggingface.co/distilgpt2
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- PEFT Library: https://github.com/huggingface/peft

### **Financial Data Sources**
- Microsoft Investor Relations: https://investor.microsoft.com/
- SEC EDGAR Database: https://www.sec.gov/edgar/searchedgar/companysearch

---

## ✅ **Assignment Completion Checklist**

- [x] **Data Collection & Preprocessing**: Microsoft 10-K reports (2022-2023)
- [x] **50 Q&A Pairs**: Comprehensive financial dataset created
- [x] **RAG System**: Hybrid retrieval (BM25 + Dense vectors)
- [x] **Advanced RAG Technique**: Hybrid Search (Sparse + Dense)
- [x] **Fine-tuned System**: DistilGPT-2 with LoRA adaptation
- [x] **Advanced Fine-tuning**: Adapter-Based Parameter-Efficient Tuning
- [x] **Guardrails**: Input/output validation implemented
- [x] **User Interface**: Streamlit web application
- [x] **Mandatory Tests**: 3 question types implemented
- [x] **Extended Evaluation**: 15+ questions evaluated
- [x] **Results Table**: Assignment-compliant format
- [x] **Performance Analysis**: Comprehensive comparison
- [x] **Documentation**: Technical implementation details
- [x] **Open-source Compliance**: No proprietary APIs used

---

## 🎓 **Academic Contributions**

This project demonstrates:
1. **Practical Implementation** of state-of-the-art RAG and fine-tuning techniques
2. **Innovative Approach** combining hybrid retrieval with parameter-efficient adaptation
3. **Comprehensive Evaluation** framework for financial domain QA systems
4. **Open-source Development** contributing to the AI research community
5. **Real-world Application** solving practical financial analysis challenges

---

**Group 46** | **Conversational AI Assignment 2** | **2024** 