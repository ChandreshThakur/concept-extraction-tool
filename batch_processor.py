"""
Batch processing utility for concept extraction across multiple subjects and formats.
"""
import os
import sys
import pandas as pd
import json
from typing import List, Dict
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csv_reader import read_questions_csv
from concept_extractor import ConceptExtractor
from simulated_llm import SimulatedLLM

class BatchProcessor:
    """
    Utility class for batch processing multiple subjects and generating reports.
    """
    
    def __init__(self, output_dir: str = "batch_output"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_all_subjects(self, subjects: List[str] = None, use_llm: bool = False) -> Dict:
        """
        Process concept extraction for all available subjects.
        """
        if subjects is None:
            subjects = self._discover_subjects()
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for subject in subjects:
            try:
                self.logger.info(f"Processing subject: {subject}")
                result = self._process_single_subject(subject, use_llm)
                results[subject] = result
                
                # Save individual subject results
                output_file = os.path.join(self.output_dir, f"{subject}_concepts_{timestamp}.csv")
                result['dataframe'][['Question Number', 'Question', 'Concepts']].to_csv(
                    output_file, index=False, encoding='utf-8'
                )
                
            except Exception as e:
                self.logger.error(f"Error processing {subject}: {e}")
                results[subject] = {'error': str(e)}
        
        # Generate summary report
        self._generate_summary_report(results, timestamp)
        
        return results
    
    def _discover_subjects(self) -> List[str]:
        """
        Discover available subjects from the resources directory.
        """
        subjects = []
        resources_dir = "resources"
        
        if os.path.exists(resources_dir):
            for file in os.listdir(resources_dir):
                if file.endswith('.csv'):
                    subject = file.replace('.csv', '')
                    subjects.append(subject)
        
        return subjects
    
    def _process_single_subject(self, subject: str, use_llm: bool = False) -> Dict:
        """
        Process a single subject and return results with statistics.
        """
        questions_file = f"resources/{subject}.csv"
        custom_dict_file = f"dictionaries/{subject}_concepts.csv"
        
        # Read questions
        questions_df = read_questions_csv(questions_file)
        
        if questions_df.empty:
            raise ValueError(f"No questions found for subject: {subject}")
        
        start_time = pd.Timestamp.now()
        
        if use_llm:
            extractor = SimulatedLLM()
            questions_df["Concepts"] = questions_df["Question"].apply(extractor.extract_concepts)
            questions_df["Concepts"] = questions_df["Concepts"].apply(
                lambda concepts: "; ".join(concepts) if concepts else ""
            )
        else:
            extractor = ConceptExtractor(
                custom_dict_file=custom_dict_file if os.path.exists(custom_dict_file) else None
            )
            questions_df = extractor.extract_concepts_from_dataframe(questions_df)
        
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_questions = len(questions_df)
        questions_with_concepts = len(questions_df[questions_df['Concepts'].str.len() > 0])
        
        # Concept frequency analysis
        all_concepts = []
        for concepts_str in questions_df['Concepts']:
            if concepts_str:
                all_concepts.extend(concepts_str.split('; '))
        
        from collections import Counter
        concept_freq = Counter(all_concepts)
        
        return {
            'dataframe': questions_df,
            'statistics': {
                'total_questions': total_questions,
                'questions_with_concepts': questions_with_concepts,
                'coverage_percentage': (questions_with_concepts / total_questions) * 100,
                'processing_time_seconds': processing_time,
                'unique_concepts_count': len(concept_freq),
                'most_common_concepts': dict(concept_freq.most_common(10))
            }
        }
    
    def _generate_summary_report(self, results: Dict, timestamp: str):
        """
        Generate a comprehensive summary report across all subjects.
        """
        summary = {
            'timestamp': timestamp,
            'total_subjects_processed': len(results),
            'successful_subjects': sum(1 for r in results.values() if 'statistics' in r),
            'failed_subjects': sum(1 for r in results.values() if 'error' in r),
            'subject_details': {}
        }
        
        total_questions_all = 0
        total_concepts_all = 0
        
        for subject, result in results.items():
            if 'statistics' in result:
                stats = result['statistics']
                summary['subject_details'][subject] = stats
                total_questions_all += stats['total_questions']
                total_concepts_all += stats['unique_concepts_count']
            else:
                summary['subject_details'][subject] = {'error': result.get('error', 'Unknown error')}
        
        summary['overall_statistics'] = {
            'total_questions_processed': total_questions_all,
            'total_unique_concepts': total_concepts_all,
            'average_concepts_per_subject': total_concepts_all / max(1, summary['successful_subjects'])
        }
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, f"batch_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary report saved to: {summary_file}")
        
        # Print summary to console
        print(f"\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Timestamp: {timestamp}")
        print(f"Subjects processed: {summary['successful_subjects']}/{summary['total_subjects_processed']}")
        print(f"Total questions: {summary['overall_statistics']['total_questions_processed']}")
        print(f"Total unique concepts: {summary['overall_statistics']['total_unique_concepts']}")
        
        if summary['failed_subjects'] > 0:
            print(f"\nFailed subjects:")
            for subject, details in summary['subject_details'].items():
                if 'error' in details:
                    print(f"  - {subject}: {details['error']}")

def main():
    """
    Command-line interface for batch processing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process concept extraction for multiple subjects.")
    parser.add_argument("--subjects", nargs='+', help="List of subjects to process (default: all discovered)")
    parser.add_argument("--use_llm", action="store_true", help="Use simulated LLM instead of hybrid extractor")
    parser.add_argument("--output_dir", default="batch_output", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create processor and run
    processor = BatchProcessor(args.output_dir)
    results = processor.process_all_subjects(args.subjects, args.use_llm)
    
    print(f"\nBatch processing complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
