import argparse
import os
import pandas as pd
import nltk
import logging

# Download necessary NLTK data (if not already downloaded)
def download_nltk_data():
    """Download required NLTK data with better error handling."""
    nltk_data_to_download = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for data_path, package_name in nltk_data_to_download:
        try:
            nltk.data.find(data_path)
        except LookupError:
            print(f"Downloading {package_name}...")
            try:
                nltk.download(package_name)
            except Exception as e:
                print(f"Error downloading {package_name}: {e}")

# Download NLTK data
download_nltk_data()

from csv_reader import read_questions_csv
from concept_extractor import ConceptExtractor
from simulated_llm import SimulatedLLM
from llm_interface import LLMConceptExtractor # Import for type hinting

def main():
    parser = argparse.ArgumentParser(description="Enhanced Concept Extraction from Competitive Exam Questions.")
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject of the exam questions (e.g., ancient_history, economics, mathematics, physics).")
    parser.add_argument("--use_llm", action="store_true",
                        help="Use simulated LLM for concept extraction (for testing LLM integration).")
    parser.add_argument("--output_format", type=str, choices=['csv', 'json', 'both'], default='csv',
                        help="Output format for results.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum confidence threshold for concept inclusion.")
    parser.add_argument("--max_concepts", type=int, default=10,
                        help="Maximum number of concepts to extract per question.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    parser.add_argument("--analytics", action="store_true",
                        help="Generate detailed analytics report.")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define file paths
    resources_dir = "resources"
    dictionaries_dir = "dictionaries"
    output_base = f"output_{args.subject}_concepts"

    questions_file = os.path.join(resources_dir, f"{args.subject}.csv")
    custom_dict_file = os.path.join(dictionaries_dir, f"{args.subject}_concepts.csv")

    # 1. Read questions from CSV
    logger.info(f"Reading questions from {questions_file}...")
    questions_df = read_questions_csv(questions_file)

    if questions_df.empty:
        logger.error("No questions to process. Exiting.")
        return

    # 2. Initialize Concept Extractor
    start_time = pd.Timestamp.now()
    
    if args.use_llm:
        logger.info("Using Enhanced Simulated LLM for concept extraction.")
        concept_extractor = SimulatedLLM()
        # Apply LLM extraction
        questions_df["Concepts"] = questions_df["Question"].apply(concept_extractor.extract_concepts)
        questions_df["Concepts"] = questions_df["Concepts"].apply(lambda concepts: "; ".join(concepts) if concepts else "")
        
        # Add concept count for analytics
        questions_df['Concept_Count'] = questions_df['Concepts'].apply(
            lambda x: len(x.split('; ')) if x else 0
        )
    else:
        logger.info("Using Enhanced Hybrid Concept Extractor (RAKE + Custom Dictionary + Patterns).")
        # Check if custom dictionary exists
        if not os.path.exists(custom_dict_file):
            logger.warning(f"Custom dictionary file not found for {args.subject} at {custom_dict_file}. "
                          "Proceeding without a custom dictionary for this subject.")
            custom_dict_file = None

        hybrid_extractor = ConceptExtractor(
            custom_dict_file=custom_dict_file,
            confidence_threshold=args.confidence_threshold,
            max_concepts=args.max_concepts
        )
        questions_df = hybrid_extractor.extract_concepts_from_dataframe(questions_df)

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    # 3. Save results in requested format(s)
    try:
        output_df = questions_df[["Question Number", "Question", "Concepts"]]
        
        if args.output_format in ['csv', 'both']:
            csv_file = f"{output_base}.csv"
            output_df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results saved to CSV: {csv_file}")
        
        if args.output_format in ['json', 'both']:
            json_file = f"{output_base}.json"
            if not args.use_llm:
                hybrid_extractor.export_concepts_to_json(output_df, json_file)
            else:
                # Manual JSON export for LLM results
                concepts_dict = {}
                for _, row in output_df.iterrows():
                    concepts_dict[str(row['Question Number'])] = {
                        'question': row['Question'],
                        'concepts': row['Concepts'].split('; ') if row['Concepts'] else [],
                        'concept_count': len(row['Concepts'].split('; ')) if row['Concepts'] else 0
                    }
                
                import json
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(concepts_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to JSON: {json_file}")

        # 4. Display sample results
        print(f"\n=== CONCEPT EXTRACTION RESULTS ===")
        print(f"Subject: {args.subject.replace('_', ' ').title()}")
        print(f"Method: {'Enhanced Simulated LLM' if args.use_llm else 'Enhanced Hybrid Extractor'}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        print(f"\n--- Sample Extracted Concepts ---")
        sample_df = output_df.head(3)
        for _, row in sample_df.iterrows():
            print(f"\nQ{row['Question Number']}: {row['Question'][:100]}...")
            print(f"Concepts: {row['Concepts']}")
        
        # 5. Generate analytics
        total_questions = len(output_df)
        questions_with_concepts = len(output_df[output_df['Concepts'].str.len() > 0])
        
        if 'Concept_Count' in questions_df.columns:
            avg_concepts = questions_df['Concept_Count'].mean()
            max_concepts = questions_df['Concept_Count'].max()
            
            print(f"\n--- Analytics Summary ---")
            print(f"Total questions processed: {total_questions}")
            print(f"Questions with extracted concepts: {questions_with_concepts}")
            print(f"Coverage: {questions_with_concepts/total_questions*100:.1f}%")
            print(f"Average concepts per question: {avg_concepts:.1f}")
            print(f"Maximum concepts in single question: {max_concepts}")
        
        # 6. Detailed analytics if requested
        if args.analytics:
            print(f"\n--- Detailed Analytics ---")
            
            # Concept frequency analysis
            all_concepts = []
            for concepts_str in output_df['Concepts']:
                if concepts_str:
                    all_concepts.extend(concepts_str.split('; '))
            
            from collections import Counter
            concept_freq = Counter(all_concepts)
            
            print(f"Total unique concepts extracted: {len(concept_freq)}")
            print(f"Most frequent concepts:")
            for concept, freq in concept_freq.most_common(5):
                print(f"  - {concept}: {freq} times")
            
            # Save detailed analytics
            analytics_file = f"analytics_{args.subject}.json"
            analytics_data = {
                'subject': args.subject,
                'extraction_method': 'Enhanced Simulated LLM' if args.use_llm else 'Enhanced Hybrid Extractor',
                'processing_time_seconds': processing_time,
                'total_questions': total_questions,
                'questions_with_concepts': questions_with_concepts,
                'coverage_percentage': questions_with_concepts/total_questions*100,
                'concept_frequency': dict(concept_freq.most_common(10)),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if 'Concept_Count' in questions_df.columns:
                analytics_data.update({
                    'avg_concepts_per_question': float(questions_df['Concept_Count'].mean()),
                    'max_concepts_single_question': int(questions_df['Concept_Count'].max()),
                    'min_concepts_single_question': int(questions_df['Concept_Count'].min())
                })
            
            import json
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False)
            print(f"Detailed analytics saved to: {analytics_file}")
        
        # 7. Get extractor statistics if using hybrid method
        if not args.use_llm:
            stats = hybrid_extractor.get_extraction_statistics()
            print(f"\n--- Extraction Statistics ---")
            print(f"Total concepts extracted: {stats['concepts_extracted']}")
            print(f"Average concepts per question: {stats['avg_concepts_per_question']:.2f}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return

    print(f"\n=== PROCESSING COMPLETE ===")
    
if __name__ == "__main__":
    main()


