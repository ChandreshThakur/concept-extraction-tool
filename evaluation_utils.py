"""
Evaluation and benchmarking utilities for concept extraction.
"""
import pandas as pd
import json
import os
from typing import List, Dict, Tuple
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class ConceptExtractionEvaluator:
    """
    Utility class for evaluating and benchmarking concept extraction performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_extraction_methods(self, subject: str) -> Dict:
        """
        Compare hybrid vs LLM extraction methods for a given subject.
        """
        from concept_extractor import ConceptExtractor
        from simulated_llm import SimulatedLLM
        from csv_reader import read_questions_csv
        
        # Load data
        questions_file = f"resources/{subject}.csv"
        custom_dict_file = f"dictionaries/{subject}_concepts.csv"
        
        questions_df = read_questions_csv(questions_file)
        if questions_df.empty:
            raise ValueError(f"No questions found for subject: {subject}")
        
        # Extract using hybrid method
        hybrid_extractor = ConceptExtractor(
            custom_dict_file=custom_dict_file if os.path.exists(custom_dict_file) else None
        )
        hybrid_df = hybrid_extractor.extract_concepts_from_dataframe(questions_df.copy())
        
        # Extract using simulated LLM
        llm_extractor = SimulatedLLM()
        llm_df = questions_df.copy()
        llm_df["Concepts"] = llm_df["Question"].apply(llm_extractor.extract_concepts)
        llm_df["Concepts"] = llm_df["Concepts"].apply(lambda concepts: "; ".join(concepts) if concepts else "")
        
        # Compare results
        comparison = self._analyze_extraction_differences(hybrid_df, llm_df)
        comparison['subject'] = subject
        
        return comparison
    
    def _analyze_extraction_differences(self, hybrid_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict:
        """
        Analyze differences between two extraction methods.
        """
        comparison = {
            'hybrid_stats': self._calculate_stats(hybrid_df),
            'llm_stats': self._calculate_stats(llm_df),
            'overlap_analysis': {},
            'unique_concepts': {}
        }
        
        # Analyze concept overlap
        hybrid_concepts = set()
        llm_concepts = set()
        
        for concepts_str in hybrid_df['Concepts']:
            if concepts_str:
                hybrid_concepts.update(concepts_str.split('; '))
        
        for concepts_str in llm_df['Concepts']:
            if concepts_str:
                llm_concepts.update(concepts_str.split('; '))
        
        overlap = hybrid_concepts.intersection(llm_concepts)
        hybrid_unique = hybrid_concepts - llm_concepts
        llm_unique = llm_concepts - hybrid_concepts
        
        comparison['overlap_analysis'] = {
            'total_hybrid_concepts': len(hybrid_concepts),
            'total_llm_concepts': len(llm_concepts),
            'overlapping_concepts': len(overlap),
            'overlap_percentage': (len(overlap) / max(len(hybrid_concepts.union(llm_concepts)), 1)) * 100
        }
        
        comparison['unique_concepts'] = {
            'hybrid_only': list(hybrid_unique)[:10],  # Top 10 for brevity
            'llm_only': list(llm_unique)[:10]
        }
        
        return comparison
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for a concept extraction result.
        """
        total_questions = len(df)
        questions_with_concepts = len(df[df['Concepts'].str.len() > 0])
        
        all_concepts = []
        for concepts_str in df['Concepts']:
            if concepts_str:
                all_concepts.extend(concepts_str.split('; '))
        
        concept_freq = Counter(all_concepts)
        
        return {
            'total_questions': total_questions,
            'questions_with_concepts': questions_with_concepts,
            'coverage_percentage': (questions_with_concepts / total_questions) * 100,
            'total_concepts_extracted': len(all_concepts),
            'unique_concepts': len(concept_freq),
            'avg_concepts_per_question': len(all_concepts) / total_questions,
            'most_common_concepts': dict(concept_freq.most_common(5))
        }
    
    def generate_performance_report(self, subjects: List[str], output_file: str = "performance_report.json"):
        """
        Generate a comprehensive performance report across multiple subjects.
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'subjects_analyzed': len(subjects),
            'method_comparisons': {},
            'overall_insights': {}
        }
        
        for subject in subjects:
            try:
                comparison = self.compare_extraction_methods(subject)
                report['method_comparisons'][subject] = comparison
            except Exception as e:
                self.logger.error(f"Error analyzing subject {subject}: {e}")
                report['method_comparisons'][subject] = {'error': str(e)}
        
        # Calculate overall insights
        successful_comparisons = [comp for comp in report['method_comparisons'].values() 
                                if 'error' not in comp]
        
        if successful_comparisons:
            avg_hybrid_coverage = sum(comp['hybrid_stats']['coverage_percentage'] 
                                    for comp in successful_comparisons) / len(successful_comparisons)
            avg_llm_coverage = sum(comp['llm_stats']['coverage_percentage'] 
                                 for comp in successful_comparisons) / len(successful_comparisons)
            avg_overlap = sum(comp['overlap_analysis']['overlap_percentage'] 
                            for comp in successful_comparisons) / len(successful_comparisons)
            
            report['overall_insights'] = {
                'average_hybrid_coverage': avg_hybrid_coverage,
                'average_llm_coverage': avg_llm_coverage,
                'average_concept_overlap': avg_overlap,
                'recommendation': self._generate_recommendation(avg_hybrid_coverage, avg_llm_coverage, avg_overlap)
            }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Performance report saved to: {output_file}")
        return report
    
    def _generate_recommendation(self, hybrid_coverage: float, llm_coverage: float, overlap: float) -> str:
        """
        Generate recommendations based on performance metrics.
        """
        if hybrid_coverage > llm_coverage + 10:
            return "Hybrid method shows significantly better coverage. Recommended for production."
        elif llm_coverage > hybrid_coverage + 10:
            return "LLM method shows significantly better coverage. Consider LLM integration."
        elif overlap > 70:
            return "Both methods show similar results with high overlap. Choose based on cost considerations."
        else:
            return "Methods show complementary results. Consider ensemble approach combining both."
    
    def visualize_comparison(self, subject: str, save_plot: bool = True):
        """
        Create visualizations comparing extraction methods.
        """
        try:
            comparison = self.compare_extraction_methods(subject)
            
            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Concept Extraction Comparison: {subject.title()}', fontsize=16)
            
            # Coverage comparison
            methods = ['Hybrid', 'LLM']
            coverage = [comparison['hybrid_stats']['coverage_percentage'], 
                       comparison['llm_stats']['coverage_percentage']]
            
            axes[0, 0].bar(methods, coverage, color=['skyblue', 'lightcoral'])
            axes[0, 0].set_title('Coverage Percentage')
            axes[0, 0].set_ylabel('Percentage')
            
            # Concepts per question comparison
            concepts_per_q = [comparison['hybrid_stats']['avg_concepts_per_question'],
                            comparison['llm_stats']['avg_concepts_per_question']]
            
            axes[0, 1].bar(methods, concepts_per_q, color=['skyblue', 'lightcoral'])
            axes[0, 1].set_title('Average Concepts per Question')
            axes[0, 1].set_ylabel('Count')
            
            # Overlap analysis
            overlap_data = comparison['overlap_analysis']
            labels = ['Hybrid Only', 'Overlap', 'LLM Only']
            sizes = [overlap_data['total_hybrid_concepts'] - overlap_data['overlapping_concepts'],
                    overlap_data['overlapping_concepts'],
                    overlap_data['total_llm_concepts'] - overlap_data['overlapping_concepts']]
            
            axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
            axes[1, 0].set_title('Concept Overlap Analysis')
            
            # Top concepts frequency (hybrid method)
            top_concepts = list(comparison['hybrid_stats']['most_common_concepts'].keys())[:5]
            frequencies = list(comparison['hybrid_stats']['most_common_concepts'].values())[:5]
            
            axes[1, 1].barh(range(len(top_concepts)), frequencies, color='skyblue')
            axes[1, 1].set_yticks(range(len(top_concepts)))
            axes[1, 1].set_yticklabels(top_concepts)
            axes[1, 1].set_title('Most Common Concepts (Hybrid)')
            axes[1, 1].set_xlabel('Frequency')
            
            plt.tight_layout()
            
            if save_plot:
                plot_file = f"comparison_{subject}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Comparison plot saved to: {plot_file}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib/seaborn not available. Skipping visualization.")
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")

def main():
    """
    Command-line interface for evaluation utilities.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate concept extraction performance.")
    parser.add_argument("--subjects", nargs='+', default=['ancient_history', 'economics', 'mathematics', 'physics'],
                       help="Subjects to analyze")
    parser.add_argument("--compare", help="Compare methods for a specific subject")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive performance report")
    parser.add_argument("--visualize", help="Create visualizations for a specific subject")
    parser.add_argument("--output", default="performance_report.json", help="Output file for reports")
    
    args = parser.parse_args()
    
    evaluator = ConceptExtractionEvaluator()
    
    if args.compare:
        comparison = evaluator.compare_extraction_methods(args.compare)
        print(json.dumps(comparison, indent=2))
    
    if args.report:
        report = evaluator.generate_performance_report(args.subjects, args.output)
        print(f"Performance report generated: {args.output}")
    
    if args.visualize:
        evaluator.visualize_comparison(args.visualize)

if __name__ == "__main__":
    main()
