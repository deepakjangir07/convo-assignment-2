# 📊 Financial Question Answering System

A comprehensive Streamlit-based web application that combines Retrieval-Augmented Generation (RAG) with fine-tuned language models for financial document analysis and question answering.

## 🚀 Features

### 🤖 AI Models
- **Base Model**: DistilGPT-2 for general question answering
- **Fine-tuned Model**: Custom-trained DistilGPT-2 for financial domain expertise
- **Hybrid Approach**: Combines both models for optimal performance

### 🔍 Advanced Retrieval
- **Dense Retrieval**: Vector similarity search using ChromaDB
- **Sparse Retrieval**: BM25 algorithm for keyword-based search
- **RRF Fusion**: Reciprocal Rank Fusion for combining retrieval results
- **Context Chunking**: Intelligent text segmentation with overlap

### 💻 Modern Web Interface
- **Streamlit UI**: Clean, responsive web interface
- **Real-time Processing**: Live question answering with progress indicators
- **Interactive Visualizations**: Charts and metrics for evaluation results
- **File Upload**: PDF processing and knowledge base expansion
- **Export Capabilities**: Download results and data in various formats

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd financial-qa-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Place your financial documents (PDFs) in a `data/` folder
   - Or use the existing Microsoft 10-K reports if available

4. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
financial-qa-system/
├── app.py                 # Main Streamlit application
├── model_utils.py         # Core QA system and model utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data directory
│   └── processed_financials.txt
├── ft_model_distilgpt2_adapters/  # Fine-tuned model (if available)
└── ConvoAI_Assignment2.ipynb      # Original notebook
```

## 🎯 Usage

### 1. System Initialization
- Navigate to the **Home** page
- Click "Initialize System" to load models and setup retrieval
- Wait for the system to load (this may take a few minutes on first run)

### 2. Asking Questions
- Go to **Ask Questions** tab
- Enter your financial question
- Choose between Base or Fine-tuned model
- Adjust advanced parameters if needed
- Click "Get Answer" to receive results

### 3. Model Evaluation
- Use the **Evaluation** tab to compare model performance
- Select questions from the predefined set
- Run evaluation to get confidence scores and response times
- View interactive charts and download results

### 4. Data Management
- **Data Management** tab for uploading new PDFs
- Process additional documents to expand knowledge base
- Export chunks and statistics for analysis

### 5. System Settings
- **Settings** tab for model configuration
- Monitor system resources and GPU usage
- Reload models or clear session data

## 🔧 Configuration

### Model Parameters
- **Max New Tokens**: Control answer length (50-200)
- **Temperature**: Adjust creativity (0.0-1.0)
- **Top-k**: Limit token selection (1-50)

### Retrieval Settings
- **Chunk Size**: Text segmentation length (default: 400)
- **Chunk Overlap**: Overlap between chunks (default: 50)
- **Top-k Retrieval**: Number of context chunks (default: 5)

## 📊 Performance Metrics

The system tracks and displays:
- **Confidence Scores**: Model certainty in answers
- **Response Times**: Processing speed for each query
- **Context Relevance**: Quality of retrieved information
- **Model Comparison**: Base vs. fine-tuned performance

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in model loading
   - Use CPU-only mode if GPU memory is insufficient
   - Close other GPU-intensive applications

2. **Model Loading Errors**
   - Ensure all dependencies are installed
   - Check if fine-tuned model files exist
   - Verify PyTorch version compatibility

3. **Data Processing Issues**
   - Ensure PDF files are not corrupted
   - Check file permissions and paths
   - Verify sufficient disk space

### Performance Tips

- **GPU Usage**: Enable CUDA for faster inference
- **Batch Processing**: Process multiple questions together
- **Memory Management**: Clear chat history for long sessions
- **Model Caching**: Keep models loaded between queries

## 🔮 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT)
- [ ] Multi-language support
- [ ] Advanced prompt engineering
- [ ] Integration with external financial APIs
- [ ] Real-time data updates
- [ ] Collaborative annotation features

## 📚 Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: PyTorch-based language models
- **Vector Store**: ChromaDB for dense retrieval
- **Text Processing**: LangChain text splitters
- **Embeddings**: Sentence Transformers

### Models Used
- **Base Model**: `distilgpt2` (82M parameters)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficiency

### Retrieval Methods
- **Dense**: Cosine similarity on sentence embeddings
- **Sparse**: BM25 scoring on tokenized text
- **Fusion**: Reciprocal Rank Fusion (RRF) algorithm

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models and libraries
- Streamlit for the web framework
- Microsoft for the financial data used in training
- The open-source AI community for tools and inspiration

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the original notebook for implementation details

---

**Note**: This system is designed for educational and research purposes. Always verify financial information from official sources before making any decisions. 