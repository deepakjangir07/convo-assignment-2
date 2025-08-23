import torch
import re
import os
import time
import numpy as np
import pandas as pd

# Handle PyMuPDF import with fallback
PYMUPDF_AVAILABLE = False
fitz = None

try:
    import PyMuPDF
    fitz = PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF imported successfully")
except ImportError:
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
        print("✅ fitz (PyMuPDF) imported successfully")
    except ImportError:
        print("⚠️  PyMuPDF not available. PDF processing will be limited.")
        PYMUPDF_AVAILABLE = False
        fitz = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Handle SQLite dependency issue for ChromaDB
CHROMADB_AVAILABLE = False
chromadb = None

try:
    import chromadb
    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB imported successfully")
except RuntimeError as e:
    if "sqlite3" in str(e):
        print("⚠️  ChromaDB not available due to SQLite version. Using alternative storage.")
        CHROMADB_AVAILABLE = False
        chromadb = None
    else:
        print(f"⚠️  ChromaDB runtime error: {e}. Using alternative storage.")
        CHROMADB_AVAILABLE = False
        chromadb = None
except ImportError as e:
    print(f"⚠️  ChromaDB not available: {e}. Using alternative storage.")
    CHROMADB_AVAILABLE = False
    chromadb = None
except Exception as e:
    print(f"⚠️  Unexpected error importing ChromaDB: {e}. Using alternative storage.")
    CHROMADB_AVAILABLE = False
    chromadb = None

from rank_bm25 import BM25Okapi
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class FinancialQASystem:
    def __init__(self, model_path="./ft_model_distilgpt2_adapters", data_path="data/processed_financials.txt"):
        self.model_path = model_path
        self.data_path = data_path
        self.chunks = None
        self.embedding_model = None
        self.collection = None
        self.bm25 = None
        self.base_model = None
        self.ft_model = None
        self.tokenizer = None
        
        # Initialize components
        self._load_models()
        self._setup_retrieval()
    
    def _load_models(self):
        """Load all necessary models and tokenizers."""
        print("Loading models...")
        
        # Load RAG model and tokenizer
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
        # Load fine-tuned model if available
        try:
            if os.path.exists(self.model_path):
                self.ft_model = PeftModel.from_pretrained(self.base_model, self.model_path)
                self.ft_model = self.ft_model.merge_and_unload()
                print("Fine-tuned model loaded successfully!")
            else:
                print("Fine-tuned model not found, using RAG model only.")
                self.ft_model = None
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            self.ft_model = None
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Models loaded successfully!")
    
    def _load_from_local_pdfs(self):
        """Load financial data from local PDF files."""
        if not PYMUPDF_AVAILABLE:
            print("❌ PyMuPDF not available. Cannot process PDFs.")
            return "PDF processing not available. Please install PyMuPDF."
        
        try:
            pdf_path_2023 = 'MSFT_2023_10K.pdf'
            pdf_path_2022 = 'MSFT_2022_10K.pdf'
            
            full_text = ""
            
            # Load 2023 PDF
            if os.path.exists(pdf_path_2023):
                print(f"Loading {pdf_path_2023}...")
                doc_2023 = fitz.open(pdf_path_2023)
                text_2023 = ""
                for page in doc_2023:
                    text_2023 += page.get_text()
                doc_2023.close()
                full_text += "--- 2023 Microsoft 10-K Report ---\n" + text_2023 + "\n\n"
                print("✅ 2023 PDF loaded successfully")
            else:
                print(f"⚠️  {pdf_path_2023} not found")
            
            # Load 2022 PDF
            if os.path.exists(pdf_path_2022):
                print(f"Loading {pdf_path_2022}...")
                doc_2022 = fitz.open(pdf_path_2022)
                text_2022 = ""
                for page in doc_2022:
                    text_2022 += page.get_text()
                doc_2022.close()
                full_text += "--- 2022 Microsoft 10-K Report ---\n" + text_2022 + "\n\n"
                print("✅ 2022 PDF loaded successfully")
            else:
                print(f"⚠️  {pdf_path_2022} not found")
            
            if full_text:
                # Save processed text locally for future use
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                with open(self.data_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"✅ Processed text saved to {self.data_path}")
                return full_text
            else:
                print("❌ No PDFs could be loaded")
                return "Microsoft financial data not available. Please ensure PDFs are accessible."
                
        except Exception as e:
            print(f"❌ Error loading PDFs: {e}")
            return f"Error loading financial data: {str(e)}"
    
    def _setup_alternative_storage(self):
        """Setup alternative storage when ChromaDB is not available."""
        print("🔄 Setting up alternative storage...")
        try:
            # Store embeddings in memory as numpy arrays
            chunk_embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=False)
            self.chunk_embeddings = chunk_embeddings
            self.collection = None  # No ChromaDB collection
            print("✅ Alternative storage (in-memory embeddings) created successfully")
        except Exception as e:
            print(f"❌ Error setting up alternative storage: {e}")
            self.chunk_embeddings = None
            self.collection = None
    
    def _setup_retrieval(self):
        """Setup the retrieval system with text chunks and indices."""
        print("Setting up retrieval system...")
        
        # Load and process text if available
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Create text chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            self.chunks = text_splitter.split_text(full_text)
            
            # Setup ChromaDB
            client = chromadb.Client()
            self.collection = client.get_or_create_collection(name="financials_rag")
            
            # Create embeddings
            chunk_embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=False)
            self.collection.add(
                ids=[str(i) for i in range(len(self.chunks))],
                embeddings=chunk_embeddings.tolist(),
                documents=self.chunks
            )
            
            # Setup BM25
            try:
                tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
                self.bm25 = BM25Okapi(tokenized_chunks)
                print("✅ BM25 sparse index created successfully")
            except Exception as e:
                print(f"⚠️  BM25 setup failed: {e}")
                self.bm25 = None
            
            print(f"✅ Retrieval system setup complete with {len(self.chunks)} chunks.")
            print(f"📊 Storage: {'ChromaDB' if self.collection else 'Alternative (In-Memory)'}")
            print(f"📊 Sparse: {'BM25' if self.bm25 else 'Not available'}")
        else:
            # Try to load from local PDFs if local data not available
            print("Local data not found, attempting to load from local PDFs...")
            full_text = self._load_from_local_pdfs()
            
            if full_text and "Error" not in full_text:
                # Create text chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                self.chunks = text_splitter.split_text(full_text)
                
                # Setup storage (ChromaDB or alternative)
                if CHROMADB_AVAILABLE and chromadb is not None:
                    try:
                        print("🔄 Attempting to setup ChromaDB...")
                        client = chromadb.Client()
                        self.collection = client.get_or_create_collection(name="financials_rag")
                        
                        # Create embeddings
                        chunk_embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=False)
                        self.collection.add(
                            ids=[str(i) for i in range(len(self.chunks))],
                            embeddings=chunk_embeddings.tolist(),
                            documents=self.chunks
                        )
                        print("✅ ChromaDB vector store created successfully")
                    except Exception as e:
                        print(f"⚠️  ChromaDB failed: {e}. Using alternative storage.")
                        self._setup_alternative_storage()
                else:
                    print("🔄 ChromaDB not available, using alternative storage...")
                    self._setup_alternative_storage()
                
                # Setup BM25
                try:
                    tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
                    self.bm25 = BM25Okapi(tokenized_chunks)
                    print("✅ BM25 sparse index created successfully")
                except Exception as e:
                    print(f"⚠️  BM25 setup failed: {e}")
                    self.bm25 = None
                
                print(f"✅ Retrieval system setup complete with {len(self.chunks)} chunks.")
                print(f"📊 Storage: {'ChromaDB' if self.collection else 'Alternative (In-Memory)'}")
                print(f"📊 Sparse: {'BM25' if self.bm25 else 'Not available'}")
            else:
                print("Data file not found and Google Drive PDFs not accessible. Please ensure data is available.")
    
    def hybrid_retrieval(self, query, top_k=5):
        """Perform hybrid retrieval using BM25 and vector search."""
        if not self.chunks:
            return ["No context available. Please ensure data is loaded."]
        
        processed_query = query.lower()
        
        # Dense Retrieval
        if self.collection and CHROMADB_AVAILABLE and chromadb is not None:
            # Use ChromaDB if available
            try:
                query_embedding = self.embedding_model.encode(processed_query).tolist()
                dense_results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
                dense_docs = dense_results['documents'][0]
                print("✅ ChromaDB retrieval successful")
            except Exception as e:
                print(f"⚠️  ChromaDB query failed: {e}. Using alternative retrieval.")
                dense_docs = self._alternative_dense_retrieval(query, top_k)
        else:
            # Use alternative storage
            print("🔄 Using alternative dense retrieval...")
            dense_docs = self._alternative_dense_retrieval(query, top_k)
        
        # Sparse Retrieval
        if self.bm25 is not None:
            try:
                tokenized_query = processed_query.split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
                sparse_docs = [self.chunks[i] for i in top_n_indices]
                print("✅ BM25 sparse retrieval successful")
            except Exception as e:
                print(f"⚠️  BM25 retrieval failed: {e}. Using simple keyword matching.")
                sparse_docs = self._simple_keyword_retrieval(query, top_k)
        else:
            print("🔄 BM25 not available, using simple keyword matching...")
            sparse_docs = self._simple_keyword_retrieval(query, top_k)
        
        # RRF Fusion
        fused_scores = {}
        k = 60
        for i, doc in enumerate(dense_docs):
            fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (k + i + 1)
        for i, doc in enumerate(sparse_docs):
            fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (k + i + 1)
        
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc for doc, score in reranked_results][:top_k]
    
    def _simple_keyword_retrieval(self, query, top_k=5):
        """Simple keyword-based retrieval when BM25 is not available."""
        try:
            query_words = query.lower().split()
            chunk_scores = []
            
            for i, chunk in enumerate(self.chunks):
                chunk_lower = chunk.lower()
                score = sum(1 for word in query_words if word in chunk_lower)
                chunk_scores.append((score, i))
            
            # Sort by score and get top-k
            chunk_scores.sort(reverse=True)
            top_indices = [i for score, i in chunk_scores[:top_k] if score > 0]
            
            if top_indices:
                return [self.chunks[i] for i in top_indices]
            else:
                # Fallback to first few chunks if no matches
                return self.chunks[:top_k]
                
        except Exception as e:
            print(f"⚠️  Simple keyword retrieval failed: {e}")
            # Ultimate fallback
            return self.chunks[:top_k] if self.chunks else []
    
    def _alternative_dense_retrieval(self, query, top_k=5):
        """Alternative dense retrieval using in-memory embeddings."""
        if self.chunk_embeddings is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities with all chunks
            similarities = np.dot(self.chunk_embeddings, query_embedding) / (
                np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [self.chunks[i] for i in top_indices]
        except Exception as e:
            print(f"⚠️  Alternative dense retrieval failed: {e}")
            return []
    
    def generate_answer(self, query, context, use_fine_tuned=True):
        """Generate answer using the specified model."""
        model_to_use = self.ft_model if use_fine_tuned and self.ft_model else self.base_model
        
        prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                num_return_sequences=1
            )
        end_time = time.time()
        
        # Decode only the newly generated tokens
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        response_time = end_time - start_time
        
        return answer, response_time
    
    def answer_query_rag(self, query, use_fine_tuned=True):
        """Complete RAG pipeline."""
        retrieved_chunks = self.hybrid_retrieval(query)
        context = "\n\n".join(retrieved_chunks)
        answer, response_time = self.generate_answer(query, context, use_fine_tuned)
        
        # Simple confidence scoring
        confidence = 0.9 if answer and len(answer) > 10 else 0.4
        
        return {
            'answer': answer,
            'confidence': confidence,
            'response_time': response_time,
            'context': context[:500] + "..." if len(context) > 500 else context
        }
    
    def process_pdf(self, pdf_path):
        """Process a PDF file and add to the knowledge base."""
        if not PYMUPDF_AVAILABLE:
            return False, "PDF processing not available. Please install PyMuPDF."
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Clean text
            text = re.sub(r'Page \d+ of \d+', '', text)
            text = re.sub(r'\s*\n\s*', '\n', text)
            text = re.sub(r'Microsoft Corporation\s+Form 10-K', '', text, flags=re.IGNORECASE)
            text = text.strip()
            
            # Update chunks and indices
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            new_chunks = text_splitter.split_text(text)
            
            if self.chunks is None:
                self.chunks = new_chunks
            else:
                self.chunks.extend(new_chunks)
            
            # Update ChromaDB
            if self.collection:
                new_embeddings = self.embedding_model.encode(new_chunks, show_progress_bar=False)
                start_id = len(self.chunks) - len(new_chunks)
                self.collection.add(
                    ids=[str(i) for i in range(start_id, len(self.chunks))],
                    embeddings=new_embeddings.tolist(),
                    documents=new_chunks
                )
            
            # Update BM25
            if self.bm25:
                tokenized_new_chunks = [chunk.lower().split() for chunk in new_chunks]
                if self.chunks:
                    all_tokenized = [chunk.lower().split() for chunk in self.chunks]
                    self.bm25 = BM25Okapi(all_tokenized)
            
            return True, f"Successfully processed {len(new_chunks)} new chunks"
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"

# Global instance
qa_system = None

def get_qa_system():
    """Get or create the QA system instance."""
    global qa_system
    if qa_system is None:
        qa_system = FinancialQASystem()
    return qa_system 