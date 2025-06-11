import pandas as pd
import nltk
from rake_nltk import Rake
from typing import List, Dict, Set, Tuple
import os
import re
import logging
from collections import Counter
import json

class ConceptExtractor:
    """
    An enhanced hybrid concept extraction system using RAKE, custom dictionaries,
    and advanced text processing techniques.
    """

    def __init__(self, stop_words_file: str = None, custom_dict_file: str = None, 
                 confidence_threshold: float = 0.5, max_concepts: int = 10):
        """
        Initializes the ConceptExtractor with enhanced capabilities.

        Args:
            stop_words_file (str, optional): Path to a custom stop words file.
            custom_dict_file (str, optional): Path to a custom keyword dictionary CSV file.
            confidence_threshold (float): Minimum confidence score for concept inclusion.
            max_concepts (int): Maximum number of concepts to extract per question.
        """
        self.stop_words = self._load_stop_words(stop_words_file)
        self.rake = Rake(stopwords=self.stop_words)
        self.custom_dictionary = self._load_custom_dictionary(custom_dict_file)
        self.confidence_threshold = confidence_threshold
        self.max_concepts = max_concepts
        
        # Enhanced features
        self.concept_patterns = self._build_concept_patterns()
        self.concept_weights = self._calculate_concept_weights()
        self.extraction_stats = {"total_questions": 0, "concepts_extracted": 0, "avg_concepts_per_question": 0}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_stop_words(self, stop_words_file: str) -> List[str]:
        """
        Loads custom stop words from a file or uses NLTK defaults.
        """
        if stop_words_file and os.path.exists(stop_words_file):
            with open(stop_words_file, 'r') as f:
                custom_stop_words = [line.strip().lower() for line in f if line.strip()]
            return nltk.corpus.stopwords.words('english') + custom_stop_words
        else:
            return nltk.corpus.stopwords.words('english')

    def _load_custom_dictionary(self, custom_dict_file: str) -> Dict[str, str]:
        """
        Loads a custom keyword dictionary from a CSV file.
        Expected format: keyword,concept
        """
        dictionary = {}
        if custom_dict_file and os.path.exists(custom_dict_file):
            try:
                df = pd.read_csv(custom_dict_file)
                if 'keyword' in df.columns and 'concept' in df.columns:
                    for index, row in df.iterrows():
                        if pd.notna(row['keyword']) and pd.notna(row['concept']):
                             dictionary[str(row['keyword']).strip().lower()] = str(row['concept']).strip()
                else:
                    print(f"Warning: Custom dictionary file {custom_dict_file} should have 'keyword' and 'concept' columns.")
            except Exception as e:
                print(f"Error loading custom dictionary {custom_dict_file}: {e}")
        return dictionary

    def extract_concepts(self, question_text: str) -> List[str]:
        """
        Enhanced concept extraction using multiple strategies with confidence scoring.
        """
        if not question_text or pd.isna(question_text):
            return []
            
        question_text = str(question_text)
        all_concepts_with_scores = []

        # 1. Extract using custom dictionary with fuzzy matching
        dict_concepts = self._fuzzy_dictionary_match(question_text)
        all_concepts_with_scores.extend(dict_concepts)

        # 2. Extract using predefined patterns
        pattern_concepts = self._extract_pattern_concepts(question_text)
        all_concepts_with_scores.extend(pattern_concepts)

        # 3. Enhanced RAKE extraction
        rake_concepts = self._enhanced_rake_extraction(question_text)
        all_concepts_with_scores.extend(rake_concepts)

        # 4. Calculate final scores and filter
        final_concepts_with_scores = self._calculate_final_scores(all_concepts_with_scores)
        
        # 5. Filter by confidence threshold and return
        filtered_concepts = [
            concept for concept, score in final_concepts_with_scores 
            if score >= self.confidence_threshold
        ]
        
        # Update statistics
        self.extraction_stats["concepts_extracted"] += len(filtered_concepts)
        
        # If no concepts meet threshold, return top 2 concepts if available
        if not filtered_concepts and final_concepts_with_scores:
            filtered_concepts = [concept for concept, _ in final_concepts_with_scores[:2]]
        
        return filtered_concepts

    def extract_concepts_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced concept extraction from a DataFrame with progress tracking and analytics.
        """
        if 'Question' not in dataframe.columns:
            self.logger.error("DataFrame must contain a 'Question' column.")
            return dataframe

        self.logger.info(f"Starting concept extraction for {len(dataframe)} questions...")
        
        # Reset statistics
        self.extraction_stats["total_questions"] = len(dataframe)
        self.extraction_stats["concepts_extracted"] = 0

        # Extract concepts with progress tracking
        dataframe['Concepts'] = dataframe['Question'].apply(self.extract_concepts)
        
        # Join concepts with semicolon for the specified output format
        dataframe['Concepts'] = dataframe['Concepts'].apply(
            lambda concepts: "; ".join(concepts) if concepts else ""
        )
        
        # Add concept count for analytics
        dataframe['Concept_Count'] = dataframe['Concepts'].apply(
            lambda x: len(x.split('; ')) if x else 0
        )
        
        # Update final statistics
        self.extraction_stats["avg_concepts_per_question"] = (
            self.extraction_stats["concepts_extracted"] / self.extraction_stats["total_questions"]
        )
        
        self.logger.info(f"Extraction complete. Statistics: {self.get_extraction_statistics()}")
        
        return dataframe

    def _build_concept_patterns(self) -> Dict[str, List[str]]:
        """
        Build regex patterns for advanced concept recognition.
        """
        patterns = {
            'historical_periods': [
                r'\b(\d+(?:st|nd|rd|th)?\s+century)\b',
                r'\b(ancient|medieval|modern)\s+period\b',
                r'\b(harappan|mauryan|gupta|mughal|british)\s+(?:period|era|empire)\b'
            ],
            'scientific_concepts': [
                r'\b(law|principle|theorem|theory)\s+of\s+\w+\b',
                r'\b\w+(?:\'s)?\s+(law|principle|theorem|theory)\b',
                r'\b(speed|velocity|acceleration|force|energy|power)\s+of\s+\w+\b'
            ],
            'mathematical_concepts': [
                r'\b(derivative|integral|equation|formula|function)\s+of\s+\w+\b',
                r'\b\w+\s+(equation|formula|function|theorem)\b',
                r'\b(area|volume|perimeter|surface)\s+of\s+\w+\b'
            ],
            'economic_concepts': [
                r'\b(law|principle|theory)\s+of\s+\w+\b',
                r'\b(monetary|fiscal|trade)\s+policy\b',
                r'\b(supply|demand|market|price)\s+\w+\b'
            ]
        }
        return patterns

    def _calculate_concept_weights(self) -> Dict[str, float]:
        """
        Calculate weights for different types of concepts based on frequency and importance.
        """
        weights = {}
        for concept in self.custom_dictionary.values():
            # Base weight
            base_weight = 1.0
            
            # Increase weight for specific important terms
            if any(term in concept.lower() for term in ['civilization', 'empire', 'period', 'law', 'theory']):
                base_weight += 0.5
            
            # Decrease weight for overly general terms
            if any(term in concept.lower() for term in ['general', 'basic', 'simple']):
                base_weight -= 0.3
                
            weights[concept] = max(0.1, base_weight)  # Minimum weight of 0.1
        
        return weights

    def _extract_pattern_concepts(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract concepts using predefined patterns with confidence scores.
        """
        concepts_with_scores = []
        text_lower = text.lower()
        
        for category, patterns in self.concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    concept = match.group().strip().title()
                    confidence = 0.8  # High confidence for pattern matches
                    concepts_with_scores.append((concept, confidence))
        
        return concepts_with_scores

    def _fuzzy_dictionary_match(self, text: str) -> List[Tuple[str, float]]:
        """
        Perform fuzzy matching against the custom dictionary.
        """
        concepts_with_scores = []
        text_lower = text.lower()
        
        for keyword, concept in self.custom_dictionary.items():
            # Exact match
            if keyword in text_lower:
                confidence = 0.9
                concepts_with_scores.append((concept, confidence))
            # Partial match (at least 70% of keyword found)
            elif len(keyword) > 4:
                words_in_keyword = keyword.split()
                matches = sum(1 for word in words_in_keyword if word in text_lower)
                if matches >= len(words_in_keyword) * 0.7:
                    confidence = 0.6 + (matches / len(words_in_keyword)) * 0.2
                    concepts_with_scores.append((concept, confidence))
        
        return concepts_with_scores

    def _enhanced_rake_extraction(self, text: str) -> List[Tuple[str, float]]:
        """
        Enhanced RAKE extraction with scoring and filtering.
        """
        try:
            self.rake.extract_keywords_from_text(text)
            rake_phrases = self.rake.get_ranked_phrases_with_scores()
            
            # Filter and score RAKE phrases
            filtered_concepts = []
            for score, phrase in rake_phrases[:15]:  # Take top 15 RAKE phrases
                # Filter out single words and very short phrases
                if len(phrase.split()) >= 2 and len(phrase) > 3:
                    # Normalize RAKE score to 0-1 range
                    normalized_score = min(1.0, score / 10.0)
                    filtered_concepts.append((phrase.title(), normalized_score))
            
            return filtered_concepts
        except Exception as e:
            self.logger.warning(f"RAKE extraction failed: {e}")
            return []

    def _calculate_final_scores(self, concepts_with_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Calculate final concept scores considering weights and deduplication.
        """
        concept_scores = {}
        
        for concept, score in concepts_with_scores:
            concept_clean = concept.strip()
            weight = self.concept_weights.get(concept_clean, 1.0)
            final_score = score * weight
            
            # If concept already exists, take the maximum score
            if concept_clean in concept_scores:
                concept_scores[concept_clean] = max(concept_scores[concept_clean], final_score)
            else:
                concept_scores[concept_clean] = final_score
        
        # Sort by score and return top concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:self.max_concepts]

    def get_extraction_statistics(self) -> Dict:
        """
        Get statistics about the extraction process.
        """
        if self.extraction_stats["total_questions"] > 0:
            self.extraction_stats["avg_concepts_per_question"] = (
                self.extraction_stats["concepts_extracted"] / self.extraction_stats["total_questions"]
            )
        return self.extraction_stats

    def export_concepts_to_json(self, concepts_df: pd.DataFrame, output_file: str):
        """
        Export concepts to JSON format for better integration.
        """
        concepts_dict = {}
        for _, row in concepts_df.iterrows():
            concepts_dict[str(row['Question Number'])] = {
                'question': row['Question'],
                'concepts': row['Concepts'].split('; ') if row['Concepts'] else [],
                'concept_count': len(row['Concepts'].split('; ')) if row['Concepts'] else 0
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concepts_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Concepts exported to JSON: {output_file}")


