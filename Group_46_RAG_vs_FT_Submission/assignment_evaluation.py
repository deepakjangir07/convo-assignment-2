"""
Assignment 2 Evaluation Script
Comprehensive evaluation of RAG vs Fine-tuned models for Financial QA System
Group 46: Hybrid Search (Sparse + Dense) + Adapter-Based Parameter-Efficient Tuning
"""

import json
import pandas as pd
import time
import numpy as np
from typing import List, Dict, Tuple
from model_utils import get_qa_system
import streamlit as st

class AssignmentEvaluator:
    def __init__(self):
        self.qa_system = None
        self.evaluation_results = []
        self.mandatory_tests = []
        
    def initialize_system(self):
        """Initialize the QA system."""
        try:
            self.qa_system = get_qa_system()
            print("✅ QA System initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing QA system: {e}")
            return False
    
            def load_qa_dataset(self, filename: str = "qa_dataset.csv"):
            """Load the QA dataset from CSV."""
            try:
                dataset_df = pd.read_csv(filename)
                print(f"✅ Loaded {len(dataset_df)} Q&A pairs from {filename}")
                
                # Convert DataFrame to list of dictionaries for compatibility
                dataset = []
                for _, row in dataset_df.iterrows():
                    dataset.append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'category': 'financial',  # Default category
                        'confidence': 'high'      # Default confidence
                    })
                
                return dataset
            except FileNotFoundError:
                print(f"❌ Dataset file {filename} not found. Please ensure qa_dataset.csv is available.")
                return None
    
    def create_mandatory_test_questions(self) -> List[Dict]:
        """Create the 3 mandatory test questions as specified in the assignment."""
        mandatory_tests = [
            {
                "id": "M1",
                "question": "What was Microsoft's revenue in 2023?",
                "expected_answer": "Microsoft's total revenue in fiscal year 2023 was $211.9 billion.",
                "category": "relevant_high_confidence",
                "description": "Relevant, high-confidence: Clear fact in data",
                "ground_truth": "The company's revenue in 2023 was $211.9 billion."
            },
            {
                "id": "M2", 
                "question": "What are the primary strategic risks related to AI development?",
                "expected_answer": "Primary strategic risks related to AI development include regulatory challenges, ethical concerns, competition from other tech companies, and potential misuse of AI technology.",
                "category": "relevant_low_confidence",
                "description": "Relevant, low-confidence: Ambiguous or sparse information",
                "ground_truth": "AI development risks include regulatory challenges, ethical concerns, and competition."
            },
            {
                "id": "M3",
                "question": "What is the capital of France?",
                "expected_answer": "This question is not related to Microsoft's financial statements. The capital of France is Paris, but this information is not contained in the financial data.",
                "category": "irrelevant",
                "description": "Irrelevant: Example: 'What is the capital of France?'",
                "ground_truth": "This question is irrelevant to Microsoft's financial data."
            }
        ]
        
        self.mandatory_tests = mandatory_tests
        return mandatory_tests
    
    def evaluate_single_question(self, question_data: Dict, use_fine_tuned: bool = True) -> Dict:
        """Evaluate a single question using the specified model."""
        if not self.qa_system:
            return {"error": "QA system not initialized"}
        
        question = question_data["question"]
        expected_answer = question_data["expected_answer"]
        category = question_data["category"]
        
        try:
            # Get answer from the system
            start_time = time.time()
            result = self.qa_system.answer_query_rag(question, use_fine_tuned)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Calculate correctness (simple keyword matching for now)
            correctness = self._calculate_correctness(result["answer"], expected_answer)
            
            # Determine confidence level
            confidence_level = self._categorize_confidence(result["confidence"])
            
            return {
                "question_id": question_data.get("id", "Q"),
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": result["answer"],
                "category": category,
                "method": "Fine-tuned" if use_fine_tuned else "RAG",
                "confidence": result["confidence"],
                "confidence_level": confidence_level,
                "response_time": response_time,
                "correctness": correctness,
                "context_used": result["context"][:200] + "..." if len(result["context"]) > 200 else result["context"]
            }
            
        except Exception as e:
            return {
                "question_id": question_data.get("id", "Q"),
                "question": question,
                "error": str(e),
                "method": "Fine-tuned" if use_fine_tuned else "RAG",
                "category": category
            }
    
    def _calculate_correctness(self, model_answer: str, expected_answer: str) -> str:
        """Calculate if the answer is correct (Y/N)."""
        if not model_answer or not expected_answer:
            return "N"
        
        # Simple keyword matching - can be enhanced with more sophisticated methods
        model_keywords = set(model_answer.lower().split())
        expected_keywords = set(expected_answer.lower().split())
        
        # Check for key financial numbers
        import re
        model_numbers = set(re.findall(r'\$[\d,]+\.?\d*[BbMmKk]?', model_answer))
        expected_numbers = set(re.findall(r'\$[\d,]+\.?\d*[BbMmKk]?', expected_answer))
        
        # If numbers match, likely correct
        if model_numbers and expected_numbers and model_numbers == expected_numbers:
            return "Y"
        
        # Check for key terms
        key_terms = ["revenue", "billion", "million", "2023", "2022", "microsoft"]
        model_has_key_terms = any(term in model_answer.lower() for term in key_terms)
        expected_has_key_terms = any(term in expected_answer.lower() for term in key_terms)
        
        if model_has_key_terms and expected_has_key_terms:
            return "Y"
        
        return "N"
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels."""
        if confidence_score >= 0.8:
            return "High"
        elif confidence_score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def run_mandatory_tests(self) -> List[Dict]:
        """Run the 3 mandatory test questions on both models."""
        print("🧪 Running Mandatory Tests...")
        
        if not self.mandatory_tests:
            self.create_mandatory_test_questions()
        
        results = []
        
        for test in self.mandatory_tests:
            print(f"Testing: {test['question']}")
            
            # Test with RAG model
            rag_result = self.evaluate_single_question(test, use_fine_tuned=False)
            results.append(rag_result)
            
            # Test with Fine-tuned model
            ft_result = self.evaluate_single_question(test, use_fine_tuned=True)
            results.append(ft_result)
        
        self.evaluation_results.extend(results)
        print(f"✅ Mandatory tests completed: {len(results)} results")
        
        return results
    
    def run_extended_evaluation(self, num_questions: int = 10) -> List[Dict]:
        """Run extended evaluation on additional questions."""
        print(f"🔍 Running Extended Evaluation on {num_questions} questions...")
        
        # Load dataset
        dataset = self.load_qa_dataset()
        if not dataset:
            return []
        
        # Select questions for evaluation (mix of categories)
        high_confidence = [q for q in dataset if q['confidence'] == 'high'][:3]
        medium_confidence = [q for q in dataset if q['confidence'] == 'medium'][:3]
        low_confidence = [q for q in dataset if q['confidence'] == 'low'][:2]
        irrelevant = [q for q in dataset if q['category'] == 'irrelevant'][:2]
        
        selected_questions = high_confidence + medium_confidence + low_confidence + irrelevant
        
        results = []
        
        for question in selected_questions[:num_questions]:
            print(f"Evaluating: {question['question'][:50]}...")
            
            # Test with RAG model
            rag_result = self.evaluate_single_question(question, use_fine_tuned=False)
            results.append(rag_result)
            
            # Test with Fine-tuned model
            ft_result = self.evaluate_single_question(question, use_fine_tuned=True)
            results.append(ft_result)
        
        self.evaluation_results.extend(results)
        print(f"✅ Extended evaluation completed: {len(results)} results")
        
        return results
    
    def create_results_table(self) -> pd.DataFrame:
        """Create the results table as specified in the assignment."""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.evaluation_results)
        
        # Clean up the data
        df = df[df['error'].isna()]  # Remove error rows
        
        # Select and rename columns for the assignment table
        results_table = df[[
            'question', 'method', 'model_answer', 'confidence', 'response_time', 'correctness'
        ]].copy()
        
        results_table.columns = ['Question', 'Method', 'Answer', 'Confidence', 'Time (s)', 'Correct (Y/N)']
        
        # Round confidence and time
        results_table['Confidence'] = results_table['Confidence'].round(2)
        results_table['Time (s)'] = results_table['Time (s)'].round(3)
        
        return results_table
    
    def generate_evaluation_summary(self) -> Dict:
        """Generate comprehensive evaluation summary."""
        if not self.evaluation_results:
            return {}
        
        df = pd.DataFrame(self.evaluation_results)
        df = df[df['error'].isna()]  # Remove error rows
        
        summary = {}
        
        # Overall statistics
        summary['total_questions'] = len(df['question'].unique())
        summary['total_responses'] = len(df)
        
        # Method comparison
        for method in ['RAG', 'Fine-tuned']:
            method_df = df[df['method'] == method]
            
            summary[f'{method.lower()}_total'] = len(method_df)
            summary[f'{method.lower()}_avg_confidence'] = method_df['confidence'].mean()
            summary[f'{method.lower()}_avg_response_time'] = method_df['response_time'].mean()
            summary[f'{method.lower()}_correctness_rate'] = (method_df['correctness'] == 'Y').mean()
        
        # Category analysis
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            summary[f'{category}_total'] = len(cat_df)
            summary[f'{category}_avg_confidence'] = cat_df['confidence'].mean()
            summary[f'{category}_correctness_rate'] = (cat_df['correctness'] == 'Y').mean()
        
        return summary
    
    def save_evaluation_results(self, filename: str = "assignment_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        if not self.evaluation_results:
            print("❌ No results to save")
            return
        
        # Prepare data for saving
        save_data = {
            "evaluation_summary": self.generate_evaluation_summary(),
            "mandatory_tests": self.mandatory_tests,
            "all_results": self.evaluation_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation results saved to {filename}")
    
    def export_results_csv(self, filename: str = "assignment_evaluation_results.csv"):
        """Export results to CSV format."""
        if not self.evaluation_results:
            print("❌ No results to export")
            return
        
        df = pd.DataFrame(self.evaluation_results)
        df = df[df['error'].isna()]  # Remove error rows
        
        # Select relevant columns
        export_df = df[[
            'question_id', 'question', 'method', 'model_answer', 'confidence', 
            'response_time', 'correctness', 'category', 'confidence_level'
        ]].copy()
        
        export_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✅ Results exported to CSV: {filename}")
    
    def print_evaluation_summary(self):
        """Print a formatted evaluation summary."""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        summary = self.generate_evaluation_summary()
        
        print("\n" + "="*80)
        print("📊 ASSIGNMENT 2 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📈 Overall Statistics:")
        print(f"  • Total Questions Evaluated: {summary.get('total_questions', 0)}")
        print(f"  • Total Responses Generated: {summary.get('total_responses', 0)}")
        
        print(f"\n🤖 Model Performance Comparison:")
        print(f"  RAG Model:")
        print(f"    • Average Confidence: {summary.get('rag_avg_confidence', 0):.3f}")
        print(f"    • Average Response Time: {summary.get('rag_avg_response_time', 0):.3f}s")
        print(f"    • Correctness Rate: {summary.get('rag_correctness_rate', 0):.1%}")
        
        print(f"  Fine-tuned Model:")
        print(f"    • Average Confidence: {summary.get('fine-tuned_avg_confidence', 0):.3f}")
        print(f"    • Average Response Time: {summary.get('fine-tuned_avg_response_time', 0):.3f}s")
        print(f"    • Correctness Rate: {summary.get('fine-tuned_correctness_rate', 0):.1%}")
        
        print(f"\n🎯 Category Analysis:")
        for key, value in summary.items():
            if key.endswith('_total') and not key.startswith(('rag_', 'fine-tuned_')):
                category = key.replace('_total', '')
                confidence = summary.get(f'{category}_avg_confidence', 0)
                correctness = summary.get(f'{category}_correctness_rate', 0)
                print(f"  {category.replace('_', ' ').title()}:")
                print(f"    • Questions: {value}")
                print(f"    • Avg Confidence: {confidence:.3f}")
                print(f"    • Correctness Rate: {correctness:.1%}")
        
        print("\n" + "="*80)

def main():
    """Main evaluation function."""
    print("🚀 Assignment 2 Evaluation - Group 46")
    print("=" * 60)
    print("RAG: Hybrid Search (Sparse + Dense)")
    print("Fine-tuning: Adapter-Based Parameter-Efficient Tuning")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = AssignmentEvaluator()
    
    # Initialize system
    if not evaluator.initialize_system():
        print("❌ Failed to initialize QA system. Exiting.")
        return
    
    # Run mandatory tests
    print("\n🧪 Running Mandatory Tests (3 questions)...")
    mandatory_results = evaluator.run_mandatory_tests()
    
    # Run extended evaluation
    print("\n🔍 Running Extended Evaluation (10+ questions)...")
    extended_results = evaluator.run_extended_evaluation(num_questions=12)
    
    # Create results table
    print("\n📊 Creating Results Table...")
    results_table = evaluator.create_results_table()
    
    if not results_table.empty:
        print("\n📋 Results Table Preview:")
        print(results_table.head(10))
        
        # Save results
        evaluator.save_evaluation_results()
        evaluator.export_results_csv()
        
        # Print summary
        evaluator.print_evaluation_summary()
        
        print("\n✅ Evaluation complete! Check the generated files:")
        print("  • assignment_evaluation_results.json - Full results")
        print("  • assignment_evaluation_results.csv - CSV format")
    else:
        print("❌ No results generated. Check system initialization.")

if __name__ == "__main__":
    main() 