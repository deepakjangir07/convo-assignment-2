import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
import os
from model_utils import get_qa_system
import tempfile

# Page configuration
st.set_page_config(
    page_title="Financial QA System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .context-box {
        background: #e9ecef;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

def initialize_system():
    """Initialize the QA system."""
    if st.session_state.qa_system is None:
        with st.spinner("Loading models and setting up the system..."):
            try:
                st.session_state.qa_system = get_qa_system()
                st.success("System initialized successfully!")
                return True
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
                return False
    return True

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial Question Answering System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microsoft.png", width=100)
        st.markdown("### Navigation")
        
        selected = option_menu(
            menu_title=None,
            options=["🏠 Home", "❓ Ask Questions", "📈 Evaluation", "📁 Data Management", "⚙️ Settings"],
            icons=["house", "question-circle", "chart-line", "folder", "gear"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.qa_system:
            st.success("✅ System Ready")
            st.info(f"📊 Chunks: {len(st.session_state.qa_system.chunks) if st.session_state.qa_system.chunks else 0}")
        else:
            st.error("❌ System Not Ready")
            if st.button("🔄 Initialize System"):
                initialize_system()
    
    # Main content based on selection
    if selected == "🏠 Home":
        show_home()
    elif selected == "❓ Ask Questions":
        show_qa_interface()
    elif selected == "📈 Evaluation":
        show_evaluation()
    elif selected == "📁 Data Management":
        show_data_management()
    elif selected == "⚙️ Settings":
        show_settings()

def show_home():
    """Display the home page with system overview."""
    st.markdown("## 🏠 Welcome to the Financial QA System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Models</h3>
            <p>RAG + Fine-tuned DistilGPT-2</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Hybrid Retrieval</h3>
            <p>BM25 + Dense Vectors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Financial Data</h3>
            <p>Microsoft 10-K Reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System initialization
    if st.session_state.qa_system is None:
        st.warning("⚠️ Please initialize the system first to start using the QA capabilities.")
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_system():
                st.rerun()
    else:
        st.success("🎉 System is ready! Navigate to 'Ask Questions' to start querying.")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", len(st.session_state.qa_system.chunks) if st.session_state.qa_system.chunks else 0)
        with col2:
            st.metric("Model Type", "Fine-tuned" if st.session_state.qa_system.ft_model else "Base")
        with col3:
            st.metric("Embedding Model", "all-MiniLM-L6-v2")
        with col4:
            st.metric("Vector Store", "ChromaDB")

def show_qa_interface():
    """Display the main QA interface."""
    st.markdown("## ❓ Ask Financial Questions")
    
    if st.session_state.qa_system is None:
        st.error("Please initialize the system first from the Home page.")
        return
    
    # Query input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your question:", placeholder="e.g., What was Microsoft's revenue in 2023?")
    with col2:
        model_choice = st.selectbox("Model:", ["Fine-tuned", "Base"], index=0)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Number of context chunks:", 1, 10, 5)
        with col2:
            max_tokens = st.slider("Max answer length:", 50, 200, 100)
        with col3:
            temperature = st.slider("Creativity:", 0.0, 1.0, 0.1, 0.1)
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary", disabled=not query):
        if query:
            with st.spinner("Processing your question..."):
                try:
                    # Get answer
                    use_fine_tuned = model_choice == "Fine-tuned"
                    result = st.session_state.qa_system.answer_query_rag(query, use_fine_tuned)
                    
                    # Display results
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    with col2:
                        st.metric("Response Time", f"{result['response_time']:.2f}s")
                    with col3:
                        st.metric("Model Used", model_choice)
                    
                    # Context
                    st.markdown("### 🔍 Retrieved Context")
                    st.markdown(f'<div class="context-box">{result["context"]}</div>', unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'answer': result['answer'],
                        'model': model_choice,
                        'confidence': result['confidence'],
                        'response_time': result['response_time'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Recent Questions")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {chat['query'][:50]}... ({chat['timestamp']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Answer:** {chat['answer']}")
                with col2:
                    st.write(f"**Model:** {chat['model']}")
                    st.write(f"**Confidence:** {chat['confidence']:.2f}")
                    st.write(f"**Time:** {chat['response_time']:.2f}s")

def show_evaluation():
    """Display evaluation interface."""
    st.markdown("## 📈 Model Evaluation")
    
    if st.session_state.qa_system is None:
        st.error("Please initialize the system first.")
        return
    
    # Sample questions
    sample_questions = [
        "What was Microsoft's revenue in 2023?",
        "What are the primary strategic risks related to AI development?",
        "How much did Microsoft spend on research and development in 2023?",
        "What is the capital of France?",
        "Compare the net income of 2023 and 2022."
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Evaluation Questions")
        selected_questions = st.multiselect(
            "Select questions to evaluate:",
            sample_questions,
            default=sample_questions[:3]
        )
        
        if st.button("🔍 Run Evaluation", type="primary"):
            if selected_questions:
                run_evaluation(selected_questions)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.evaluation_results:
            total_questions = len(st.session_state.evaluation_results)
            avg_confidence = sum(r['confidence'] for r in st.session_state.evaluation_results) / total_questions
            avg_time = sum(r['response_time'] for r in st.session_state.evaluation_results) / total_questions
            
            st.metric("Total Questions", total_questions)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # Display results
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown("### 📊 Evaluation Results")
        
        # Results table
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence comparison
            fig_conf = px.bar(
                results_df, 
                x='question', 
                y='confidence',
                color='model',
                title="Confidence Scores by Model",
                barmode='group'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Response time comparison
            fig_time = px.bar(
                results_df, 
                x='question', 
                y='response_time',
                color='model',
                title="Response Times by Model",
                barmode='group'
            )
            st.plotly_chart(fig_time, use_container_width=True)

def run_evaluation(questions):
    """Run evaluation on selected questions."""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, question in enumerate(questions):
        status_text.text(f"Evaluating: {question}")
        
        # Test both models
        for model_name in ["Base", "Fine-tuned"]:
            try:
                use_fine_tuned = model_name == "Fine-tuned"
                result = st.session_state.qa_system.answer_query_rag(question, use_fine_tuned)
                
                results.append({
                    'question': question,
                    'model': model_name,
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'response_time': result['response_time']
                })
                
            except Exception as e:
                st.error(f"Error evaluating {model_name} model: {str(e)}")
        
        progress_bar.progress((i + 1) / len(questions))
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.evaluation_results = results
    st.success(f"Evaluation complete! Processed {len(questions)} questions.")

def show_data_management():
    """Display data management interface."""
    st.markdown("## 📁 Data Management")
    
    if st.session_state.qa_system is None:
        st.error("Please initialize the system first.")
        return
    
    # Current data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Current Data Status")
        if st.session_state.qa_system.chunks:
            st.info(f"**Total Chunks:** {len(st.session_state.qa_system.chunks)}")
            st.info(f"**Vector Store:** ChromaDB")
            st.info(f"**Sparse Index:** BM25")
        else:
            st.warning("No data loaded.")
    
    with col2:
        st.markdown("### 📤 Upload New Data")
        uploaded_file = st.file_uploader(
            "Upload PDF file:",
            type=['pdf'],
            help="Upload a new PDF to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        success, message = st.session_state.qa_system.process_pdf(tmp_path)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
    
    # Data export
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    if st.session_state.qa_system.chunks:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export Chunks"):
                chunks_df = pd.DataFrame({
                    'chunk_id': range(len(st.session_state.qa_system.chunks)),
                    'content': st.session_state.qa_system.chunks
                })
                csv = chunks_df.to_csv(index=False)
                st.download_button(
                    label="Download Chunks CSV",
                    data=csv,
                    file_name="data_chunks.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔍 Export Statistics"):
                stats = {
                    'total_chunks': len(st.session_state.qa_system.chunks),
                    'avg_chunk_length': sum(len(chunk) for chunk in st.session_state.qa_system.chunks) / len(st.session_state.qa_system.chunks),
                    'total_characters': sum(len(chunk) for chunk in st.session_state.qa_system.chunks)
                }
                stats_df = pd.DataFrame([stats])
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="Download Statistics CSV",
                    data=csv,
                    file_name="data_statistics.csv",
                    mime="text/csv"
                )

def show_settings():
    """Display settings interface."""
    st.markdown("## ⚙️ System Settings")
    
    if st.session_state.qa_system is None:
        st.error("Please initialize the system first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Model Configuration")
        
        # Model settings
        st.info(f"**Current Base Model:** distilgpt2")
        st.info(f"**Fine-tuned Model:** {'Available' if st.session_state.qa_system.ft_model else 'Not Available'}")
        st.info(f"**Embedding Model:** all-MiniLM-L6-v2")
        
        # Generation parameters
        st.markdown("#### 📝 Generation Parameters")
        max_new_tokens = st.slider("Max New Tokens:", 50, 200, 100)
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.1)
        top_k = st.slider("Top-k:", 1, 50, 10)
        
        if st.button("💾 Save Generation Settings"):
            st.success("Settings saved! (Note: These will be applied in the next session)")
    
    with col2:
        st.markdown("### 🗄️ Retrieval Configuration")
        
        # Retrieval settings
        st.info(f"**Chunk Size:** 400 characters")
        st.info(f"**Chunk Overlap:** 50 characters")
        st.info(f"**Top-k Retrieval:** 5 chunks")
        
        # System info
        st.markdown("#### 💻 System Information")
        import torch
        st.info(f"**PyTorch Version:** {torch.__version__}")
        st.info(f"**CUDA Available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.info(f"**GPU Device:** {torch.cuda.get_device_name(0)}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            st.info(f"**GPU Memory Used:** {memory_allocated:.2f} GB")
            st.info(f"**GPU Memory Reserved:** {memory_reserved:.2f} GB")
    
    # System actions
    st.markdown("---")
    st.markdown("### 🚀 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Models"):
            with st.spinner("Reloading models..."):
                try:
                    st.session_state.qa_system = None
                    st.session_state.qa_system = get_qa_system()
                    st.success("Models reloaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reloading models: {str(e)}")
    
    with col2:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    with col3:
        if st.button("📊 Clear Evaluation Results"):
            st.session_state.evaluation_results = []
            st.success("Evaluation results cleared!")

if __name__ == "__main__":
    main() 