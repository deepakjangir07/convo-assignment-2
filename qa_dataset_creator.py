"""
QA Dataset Creator for Financial QA System
Creates 50 Q&A pairs from financial data for fine-tuning and evaluation.
"""

import re
import json
import pandas as pd
from typing import List, Dict, Tuple

class FinancialQAGenerator:
    def __init__(self):
        self.qa_pairs = []
        
    def generate_revenue_questions(self) -> List[Dict]:
        """Generate questions about revenue and financial performance."""
        questions = [
            {
                "question": "What was Microsoft's total revenue in fiscal year 2023?",
                "answer": "Microsoft's total revenue in fiscal year 2023 was $211.9 billion.",
                "category": "revenue",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's total revenue in fiscal year 2022?",
                "answer": "Microsoft's total revenue in fiscal year 2022 was $198.3 billion.",
                "category": "revenue",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "How much did Microsoft's revenue increase from 2022 to 2023?",
                "answer": "Microsoft's revenue increased by $13.6 billion from 2022 to 2023, representing a 6.9% growth.",
                "category": "revenue_growth",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft Cloud revenue in 2023?",
                "answer": "Microsoft Cloud revenue in 2023 was $111.6 billion.",
                "category": "cloud_revenue",
                "confidence": "high",
                "data_source": "segment_breakdown"
            },
            {
                "question": "What percentage of total revenue did Microsoft Cloud represent in 2023?",
                "answer": "Microsoft Cloud represented approximately 52.7% of total revenue in 2023.",
                "category": "revenue_mix",
                "confidence": "high",
                "data_source": "segment_breakdown"
            }
        ]
        return questions
    
    def generate_cost_questions(self) -> List[Dict]:
        """Generate questions about costs and expenses."""
        questions = [
            {
                "question": "What was Microsoft's total cost of revenue in 2023?",
                "answer": "Microsoft's total cost of revenue in 2023 was $65.9 billion.",
                "category": "costs",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's gross margin in 2023?",
                "answer": "Microsoft's gross margin in 2023 was $146.0 billion.",
                "category": "profitability",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's gross margin percentage in 2023?",
                "answer": "Microsoft's gross margin percentage in 2023 was 68.9%.",
                "category": "profitability",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "How much did Microsoft spend on research and development in 2023?",
                "answer": "Microsoft spent $27.2 billion on research and development in 2023.",
                "category": "rd_expenses",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What were Microsoft's sales and marketing expenses in 2023?",
                "answer": "Microsoft's sales and marketing expenses in 2023 were $22.7 billion.",
                "category": "marketing_expenses",
                "confidence": "high",
                "data_source": "income_statement"
            }
        ]
        return questions
    
    def generate_profitability_questions(self) -> List[Dict]:
        """Generate questions about profitability and net income."""
        questions = [
            {
                "question": "What was Microsoft's net income in fiscal year 2023?",
                "answer": "Microsoft's net income in fiscal year 2023 was $72.4 billion.",
                "category": "net_income",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's net income in fiscal year 2022?",
                "answer": "Microsoft's net income in fiscal year 2022 was $72.7 billion.",
                "category": "net_income",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "How did Microsoft's net income change from 2022 to 2023?",
                "answer": "Microsoft's net income decreased slightly from $72.7 billion in 2022 to $72.4 billion in 2023, a decrease of $0.3 billion.",
                "category": "profitability_trend",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's operating income in 2023?",
                "answer": "Microsoft's operating income in 2023 was $88.5 billion.",
                "category": "operating_income",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "What was Microsoft's operating margin in 2023?",
                "answer": "Microsoft's operating margin in 2023 was 41.8%.",
                "category": "operating_margin",
                "confidence": "high",
                "data_source": "income_statement"
            }
        ]
        return questions
    
    def generate_balance_sheet_questions(self) -> List[Dict]:
        """Generate questions about balance sheet items."""
        questions = [
            {
                "question": "What was Microsoft's total assets as of June 30, 2023?",
                "answer": "Microsoft's total assets as of June 30, 2023 were $411.9 billion.",
                "category": "total_assets",
                "confidence": "high",
                "data_source": "balance_sheet"
            },
            {
                "question": "What was Microsoft's total liabilities as of June 30, 2023?",
                "answer": "Microsoft's total liabilities as of June 30, 2023 were $205.8 billion.",
                "category": "total_liabilities",
                "confidence": "high",
                "data_source": "balance_sheet"
            },
            {
                "question": "What was Microsoft's total stockholders' equity as of June 30, 2023?",
                "answer": "Microsoft's total stockholders' equity as of June 30, 2023 was $206.1 billion.",
                "category": "stockholders_equity",
                "confidence": "high",
                "data_source": "balance_sheet"
            },
            {
                "question": "What was Microsoft's cash and cash equivalents as of June 30, 2023?",
                "answer": "Microsoft's cash and cash equivalents as of June 30, 2023 were $34.7 billion.",
                "category": "cash_position",
                "confidence": "high",
                "data_source": "balance_sheet"
            },
            {
                "question": "What was Microsoft's total debt as of June 30, 2023?",
                "answer": "Microsoft's total debt as of June 30, 2023 was $59.7 billion.",
                "category": "debt",
                "confidence": "high",
                "data_source": "balance_sheet"
            }
        ]
        return questions
    
    def generate_cash_flow_questions(self) -> List[Dict]:
        """Generate questions about cash flow."""
        questions = [
            {
                "question": "What was Microsoft's operating cash flow in fiscal year 2023?",
                "answer": "Microsoft's operating cash flow in fiscal year 2023 was $87.6 billion.",
                "category": "operating_cash_flow",
                "confidence": "high",
                "data_source": "cash_flow_statement"
            },
            {
                "question": "What was Microsoft's investing cash flow in fiscal year 2023?",
                "answer": "Microsoft's investing cash flow in fiscal year 2023 was -$23.8 billion.",
                "category": "investing_cash_flow",
                "confidence": "high",
                "data_source": "cash_flow_statement"
            },
            {
                "question": "What was Microsoft's financing cash flow in fiscal year 2023?",
                "answer": "Microsoft's financing cash flow in fiscal year 2023 was -$58.9 billion.",
                "category": "financing_cash_flow",
                "confidence": "high",
                "data_source": "cash_flow_statement"
            },
            {
                "question": "How much did Microsoft spend on share repurchases in 2023?",
                "answer": "Microsoft spent $47.5 billion on share repurchases in 2023.",
                "category": "share_repurchases",
                "confidence": "high",
                "data_source": "cash_flow_statement"
            },
            {
                "question": "What was Microsoft's free cash flow in fiscal year 2023?",
                "answer": "Microsoft's free cash flow in fiscal year 2023 was $63.8 billion.",
                "category": "free_cash_flow",
                "confidence": "high",
                "data_source": "cash_flow_statement"
            }
        ]
        return questions
    
    def generate_segment_questions(self) -> List[Dict]:
        """Generate questions about business segments."""
        questions = [
            {
                "question": "What was the revenue from Office Commercial products and cloud services in 2023?",
                "answer": "Office Commercial products and cloud services revenue in 2023 was $44.4 billion.",
                "category": "office_commercial",
                "confidence": "high",
                "data_source": "segment_breakdown"
            },
            {
                "question": "What was the revenue from Office Consumer products and cloud services in 2023?",
                "answer": "Office Consumer products and cloud services revenue in 2023 was $5.7 billion.",
                "category": "office_consumer",
                "confidence": "high",
                "data_source": "segment_breakdown"
            },
            {
                "question": "What was the revenue from Windows in 2023?",
                "answer": "Windows revenue in 2023 was $14.7 billion.",
                "category": "windows",
                "confidence": "high",
                "data_source": "segment_breakdown"
            },
            {
                "question": "What was the revenue from Xbox content and services in 2023?",
                "answer": "Xbox content and services revenue in 2023 was $15.5 billion.",
                "category": "gaming",
                "confidence": "high",
                "data_source": "segment_breakdown"
            },
            {
                "question": "What was the revenue from LinkedIn in 2023?",
                "answer": "LinkedIn revenue in 2023 was $15.2 billion.",
                "category": "linkedin",
                "confidence": "high",
                "data_source": "segment_breakdown"
            }
        ]
        return questions
    
    def generate_risk_questions(self) -> List[Dict]:
        """Generate questions about risks and challenges."""
        questions = [
            {
                "question": "What are the primary strategic risks related to AI development?",
                "answer": "Primary strategic risks related to AI development include regulatory challenges, ethical concerns, competition from other tech companies, and potential misuse of AI technology.",
                "category": "ai_risks",
                "confidence": "medium",
                "data_source": "risk_factors"
            },
            {
                "question": "What cybersecurity risks does Microsoft face?",
                "answer": "Microsoft faces cybersecurity risks including sophisticated cyberattacks, data breaches, intellectual property theft, and the need to protect customer data and systems.",
                "category": "cybersecurity_risks",
                "confidence": "medium",
                "data_source": "risk_factors"
            },
            {
                "question": "What are the main competitive risks for Microsoft?",
                "answer": "Main competitive risks include competition from cloud providers like AWS and Google Cloud, open-source alternatives, and rapid technological changes in the industry.",
                "category": "competitive_risks",
                "confidence": "medium",
                "data_source": "risk_factors"
            },
            {
                "question": "What regulatory risks does Microsoft face?",
                "answer": "Microsoft faces regulatory risks including antitrust investigations, data privacy regulations like GDPR, and potential changes in tax laws and trade policies.",
                "category": "regulatory_risks",
                "confidence": "medium",
                "data_source": "risk_factors"
            },
            {
                "question": "What are the risks related to Microsoft's international operations?",
                "answer": "Risks include currency fluctuations, political instability, trade restrictions, and varying regulatory environments across different countries.",
                "category": "international_risks",
                "confidence": "medium",
                "data_source": "risk_factors"
            }
        ]
        return questions
    
    def generate_comparison_questions(self) -> List[Dict]:
        """Generate questions comparing different years or metrics."""
        questions = [
            {
                "question": "Compare Microsoft's revenue growth between 2022 and 2023.",
                "answer": "Microsoft's revenue grew from $198.3 billion in 2022 to $211.9 billion in 2023, representing a 6.9% increase. This growth was driven by strong performance in cloud services and productivity software.",
                "category": "year_comparison",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "How did Microsoft's profit margins change from 2022 to 2023?",
                "answer": "Microsoft's gross margin increased from 68.2% in 2022 to 68.9% in 2023, while operating margin increased from 41.5% to 41.8%. Net income margin decreased slightly from 36.7% to 34.2%.",
                "category": "margin_comparison",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "Compare Microsoft's R&D spending between 2022 and 2023.",
                "answer": "Microsoft's R&D spending increased from $24.5 billion in 2022 to $27.2 billion in 2023, representing an 11.0% increase, reflecting continued investment in innovation and new technologies.",
                "category": "rd_comparison",
                "confidence": "high",
                "data_source": "income_statement"
            },
            {
                "question": "How did Microsoft's cash position change from 2022 to 2023?",
                "answer": "Microsoft's cash and cash equivalents decreased from $104.8 billion in 2022 to $34.7 billion in 2023, primarily due to share repurchases, dividend payments, and strategic investments.",
                "category": "cash_comparison",
                "confidence": "high",
                "data_source": "balance_sheet"
            },
            {
                "question": "Compare Microsoft's debt levels between 2022 and 2023.",
                "answer": "Microsoft's total debt increased from $47.0 billion in 2022 to $59.7 billion in 2023, representing a 27.0% increase, primarily due to new debt issuances for strategic purposes.",
                "category": "debt_comparison",
                "confidence": "high",
                "data_source": "balance_sheet"
            }
        ]
        return questions
    
    def generate_irrelevant_questions(self) -> List[Dict]:
        """Generate irrelevant questions for testing robustness."""
        questions = [
            {
                "question": "What is the capital of France?",
                "answer": "This question is not related to Microsoft's financial statements. The capital of France is Paris, but this information is not contained in the financial data.",
                "category": "irrelevant",
                "confidence": "low",
                "data_source": "not_applicable"
            },
            {
                "question": "What is the weather like in Seattle today?",
                "answer": "This question is not related to Microsoft's financial statements. Weather information is not contained in the financial data.",
                "category": "irrelevant",
                "confidence": "low",
                "data_source": "not_applicable"
            },
            {
                "question": "How do you cook pasta?",
                "answer": "This question is not related to Microsoft's financial statements. Cooking instructions are not contained in the financial data.",
                "category": "irrelevant",
                "confidence": "low",
                "data_source": "not_applicable"
            },
            {
                "question": "What is the population of Tokyo?",
                "answer": "This question is not related to Microsoft's financial statements. Population data is not contained in the financial data.",
                "category": "irrelevant",
                "confidence": "low",
                "data_source": "not_applicable"
            },
            {
                "question": "What is the meaning of life?",
                "answer": "This philosophical question is not related to Microsoft's financial statements. Such information is not contained in the financial data.",
                "category": "irrelevant",
                "confidence": "low",
                "data_source": "not_applicable"
            }
        ]
        return questions
    
    def generate_ambiguous_questions(self) -> List[Dict]:
        """Generate ambiguous questions for testing low confidence scenarios."""
        questions = [
            {
                "question": "What are Microsoft's future growth prospects?",
                "answer": "Based on the financial data, Microsoft shows strong growth in cloud services and productivity software. However, future growth prospects depend on many factors not fully detailed in the financial statements, including market conditions, competition, and technological changes.",
                "category": "future_outlook",
                "confidence": "low",
                "data_source": "management_discussion"
            },
            {
                "question": "How will AI impact Microsoft's business model?",
                "answer": "While Microsoft is investing heavily in AI (evidenced by R&D spending of $27.2 billion in 2023), the specific impact on the business model is not fully detailed in the financial statements. This would require additional forward-looking analysis.",
                "category": "ai_impact",
                "confidence": "low",
                "data_source": "strategic_discussion"
            },
            {
                "question": "What is Microsoft's market share in cloud computing?",
                "answer": "The financial statements show Microsoft Cloud revenue of $111.6 billion in 2023, but do not provide specific market share data. This information would require external market research data.",
                "category": "market_share",
                "confidence": "low",
                "data_source": "external_data_required"
            },
            {
                "question": "How does Microsoft compare to its competitors financially?",
                "answer": "While Microsoft's financial performance is strong with $211.9 billion revenue and $72.4 billion net income in 2023, direct competitor comparisons are not provided in the financial statements. This would require analysis of other companies' financial data.",
                "category": "competitive_analysis",
                "confidence": "low",
                "data_source": "external_data_required"
            },
            {
                "question": "What is the long-term sustainability of Microsoft's business model?",
                "answer": "Microsoft shows strong financial fundamentals with consistent revenue growth and profitability. However, long-term sustainability depends on many external factors not fully addressed in the financial statements, including technological disruption and market changes.",
                "category": "sustainability",
                "confidence": "low",
                "data_source": "strategic_analysis"
            }
        ]
        return questions
    
    def create_full_dataset(self) -> List[Dict]:
        """Create the complete dataset with all 50 Q&A pairs."""
        all_questions = []
        
        # Add questions from each category
        all_questions.extend(self.generate_revenue_questions())
        all_questions.extend(self.generate_cost_questions())
        all_questions.extend(self.generate_profitability_questions())
        all_questions.extend(self.generate_balance_sheet_questions())
        all_questions.extend(self.generate_cash_flow_questions())
        all_questions.extend(self.generate_segment_questions())
        all_questions.extend(self.generate_risk_questions())
        all_questions.extend(self.generate_comparison_questions())
        all_questions.extend(self.generate_irrelevant_questions())
        all_questions.extend(self.generate_ambiguous_questions())
        
        # Add unique IDs
        for i, qa in enumerate(all_questions):
            qa['id'] = i + 1
            qa['difficulty'] = 'easy' if qa['confidence'] == 'high' else 'medium' if qa['confidence'] == 'medium' else 'hard'
        
        return all_questions
    
    def save_dataset(self, filename: str = "financial_qa_dataset.json"):
        """Save the dataset to a JSON file."""
        dataset = self.create_full_dataset()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved to {filename}")
        print(f"📊 Total Q&A pairs: {len(dataset)}")
        
        # Print summary by category
        categories = {}
        for qa in dataset:
            cat = qa['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📋 Questions by category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} questions")
        
        return dataset
    
    def create_csv_dataset(self, filename: str = "financial_qa_dataset.csv"):
        """Create a CSV version of the dataset for easy viewing."""
        dataset = self.create_full_dataset()
        
        # Flatten the data for CSV
        csv_data = []
        for qa in dataset:
            csv_data.append({
                'id': qa['id'],
                'question': qa['question'],
                'answer': qa['answer'],
                'category': qa['category'],
                'confidence': qa['confidence'],
                'difficulty': qa['difficulty'],
                'data_source': qa['data_source']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"✅ CSV dataset saved to {filename}")
        return df

def main():
    """Main function to create the dataset."""
    print("🚀 Creating Financial QA Dataset for Assignment 2")
    print("=" * 60)
    
    generator = FinancialQAGenerator()
    
    # Create and save the dataset
    dataset = generator.save_dataset()
    
    # Also create CSV version
    csv_df = generator.create_csv_dataset()
    
    print("\n🎯 Dataset Summary:")
    print(f"  • Total Questions: {len(dataset)}")
    print(f"  • High Confidence: {len([q for q in dataset if q['confidence'] == 'high'])}")
    print(f"  • Medium Confidence: {len([q for q in dataset if q['confidence'] == 'medium'])}")
    print(f"  • Low Confidence: {len([q for q in dataset if q['confidence'] == 'low'])}")
    print(f"  • Relevant Questions: {len([q for q in dataset if q['category'] != 'irrelevant'])}")
    print(f"  • Irrelevant Questions: {len([q for q in dataset if q['category'] == 'irrelevant'])}")
    
    print("\n📁 Files created:")
    print("  • financial_qa_dataset.json - Full dataset with metadata")
    print("  • financial_qa_dataset.csv - CSV format for easy viewing")
    
    print("\n✅ Dataset creation complete! Use these files for:")
    print("  • Fine-tuning your language model")
    print("  • RAG system evaluation")
    print("  • Assignment testing and comparison")

if __name__ == "__main__":
    main() 