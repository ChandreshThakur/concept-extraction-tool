from typing import List, Dict, Tuple
from llm_interface import LLMConceptExtractor
import re
import random
from collections import defaultdict

class SimulatedLLM(LLMConceptExtractor):
    """
    An enhanced simulated LLM for concept extraction with more sophisticated logic
    and domain-specific knowledge bases.
    """

    def __init__(self):
        """Initialize the simulated LLM with comprehensive knowledge bases."""
        self.knowledge_base = self._build_knowledge_base()
        self.concept_patterns = self._build_concept_patterns()
        self.confidence_scores = defaultdict(float)

    def _build_knowledge_base(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build a comprehensive knowledge base for different subjects.
        """
        return {
            'history': {
                'civilizations': ['Indus Valley Civilization', 'Harappan Civilization', 'Mesopotamian Civilization'],
                'empires': ['Mauryan Empire', 'Gupta Empire', 'Mughal Empire', 'British Empire', 'Roman Empire'],
                'periods': ['Ancient Period', 'Medieval Period', 'Modern Period', 'Vedic Period', 'Colonial Period'],
                'rulers': ['Ashoka', 'Chandragupta Maurya', 'Akbar', 'Shah Jahan', 'Aurangzeb'],
                'concepts': ['Land Revenue Systems', 'Village Administration', 'Temple Architecture', 
                           'Trade and Commerce', 'Social Structure', 'Religious Movements'],
                'art_culture': ['Gandhara Art', 'Mathura School', 'Temple Architecture', 'Sculpture', 
                              'Literature', 'Painting', 'Music and Dance']
            },
            'economics': {
                'theories': ['Keynesian Economics', 'Classical Economics', 'Monetarism', 'Supply-side Economics'],
                'policies': ['Monetary Policy', 'Fiscal Policy', 'Trade Policy', 'Industrial Policy'],
                'concepts': ['Inflation', 'Deflation', 'GDP', 'GNP', 'Balance of Payments', 'Exchange Rates'],
                'markets': ['Perfect Competition', 'Monopoly', 'Oligopoly', 'Monopolistic Competition'],
                'indicators': ['Consumer Price Index', 'Producer Price Index', 'Unemployment Rate', 'Interest Rates']
            },
            'mathematics': {
                'calculus': ['Differential Calculus', 'Integral Calculus', 'Limits and Continuity'],
                'algebra': ['Linear Algebra', 'Abstract Algebra', 'Polynomial Equations', 'Matrix Theory'],
                'geometry': ['Euclidean Geometry', 'Coordinate Geometry', 'Trigonometry', 'Solid Geometry'],
                'statistics': ['Probability Theory', 'Statistical Inference', 'Regression Analysis'],
                'number_theory': ['Prime Numbers', 'Number Systems', 'Modular Arithmetic']
            },
            'physics': {
                'mechanics': ['Classical Mechanics', 'Quantum Mechanics', 'Fluid Mechanics'],
                'thermodynamics': ['Laws of Thermodynamics', 'Heat Transfer', 'Kinetic Theory'],
                'electromagnetism': ['Electrostatics', 'Magnetism', 'Electromagnetic Induction', 'Maxwell Equations'],
                'optics': ['Geometrical Optics', 'Wave Optics', 'Laser Physics'],
                'modern_physics': ['Relativity Theory', 'Atomic Structure', 'Nuclear Physics', 'Particle Physics']
            }
        }

    def _build_concept_patterns(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Build patterns for concept recognition with confidence scores.
        Returns: Dict[category, List[Tuple[pattern, concept, confidence]]]
        """
        return {
            'historical_entities': [
                (r'\b(harappan|indus valley)\b', 'Indus Valley Civilization', 0.95),
                (r'\b(mauryan|chandragupta|ashoka)\b', 'Mauryan Empire', 0.90),
                (r'\b(gupta)\b', 'Gupta Period', 0.85),
                (r'\b(mughal|akbar|shah jahan|aurangzeb)\b', 'Mughal Empire', 0.90),
                (r'\b(vedic|veda)\b', 'Vedic Period', 0.85),
                (r'\b(british|colonial)\b', 'Colonial Period', 0.80)
            ],
            'economic_terms': [
                (r'\b(gdp|gross domestic product)\b', 'National Income Accounting', 0.95),
                (r'\b(inflation|price level)\b', 'Inflation and Price Theory', 0.90),
                (r'\b(monetary policy|central bank)\b', 'Monetary Policy', 0.95),
                (r'\b(fiscal policy|government spending)\b', 'Fiscal Policy', 0.95),
                (r'\b(supply and demand|market)\b', 'Market Theory', 0.85),
                (r'\b(perfect competition|monopoly)\b', 'Market Structures', 0.90)
            ],
            'mathematical_concepts': [
                (r'\b(derivative|differentiation)\b', 'Differential Calculus', 0.95),
                (r'\b(integral|integration)\b', 'Integral Calculus', 0.95),
                (r'\b(matrix|matrices)\b', 'Linear Algebra', 0.90),
                (r'\b(trigonometry|sine|cosine|tangent)\b', 'Trigonometry', 0.90),
                (r'\b(probability|statistics)\b', 'Probability and Statistics', 0.85),
                (r'\b(geometry|triangle|circle)\b', 'Geometry', 0.80)
            ],
            'physics_concepts': [
                (r'\b(newton|force|motion)\b', 'Classical Mechanics', 0.90),
                (r'\b(electric|electricity|current)\b', 'Electricity and Magnetism', 0.90),
                (r'\b(light|optics|lens)\b', 'Optics', 0.85),
                (r'\b(heat|temperature|thermodynamics)\b', 'Thermodynamics', 0.90),
                (r'\b(atom|nuclear|particle)\b', 'Modern Physics', 0.85),
                (r'\b(wave|frequency|wavelength)\b', 'Wave Physics', 0.80)
            ]
        }

    def _detect_subject_domain(self, question_text: str) -> str:
        """
        Detect the likely subject domain of the question.
        """
        text_lower = question_text.lower()
        
        # Count domain-specific keywords
        domain_scores = defaultdict(int)
        
        history_keywords = ['civilization', 'empire', 'period', 'ancient', 'medieval', 'ruler', 'dynasty']
        economics_keywords = ['economy', 'market', 'price', 'policy', 'gdp', 'inflation', 'trade']
        math_keywords = ['equation', 'formula', 'theorem', 'calculate', 'solve', 'derivative', 'integral']
        physics_keywords = ['force', 'energy', 'motion', 'electric', 'magnetic', 'wave', 'particle']
        
        for keyword in history_keywords:
            if keyword in text_lower:
                domain_scores['history'] += 1
        
        for keyword in economics_keywords:
            if keyword in text_lower:
                domain_scores['economics'] += 1
        
        for keyword in math_keywords:
            if keyword in text_lower:
                domain_scores['mathematics'] += 1
        
        for keyword in physics_keywords:
            if keyword in text_lower:
                domain_scores['physics'] += 1
        
        # Return domain with highest score, default to 'general'
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'

    def _extract_using_patterns(self, question_text: str) -> List[Tuple[str, float]]:
        """
        Extract concepts using predefined patterns.
        """
        concepts_with_confidence = []
        text_lower = question_text.lower()
        
        for category, patterns in self.concept_patterns.items():
            for pattern, concept, confidence in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    concepts_with_confidence.append((concept, confidence))
        
        return concepts_with_confidence

    def _extract_using_knowledge_base(self, question_text: str, domain: str) -> List[Tuple[str, float]]:
        """
        Extract concepts using domain-specific knowledge base.
        """
        concepts_with_confidence = []
        text_lower = question_text.lower()
        
        if domain in self.knowledge_base:
            domain_kb = self.knowledge_base[domain]
            
            for category, concepts in domain_kb.items():
                for concept in concepts:
                    # Check if any words from the concept appear in the question
                    concept_words = concept.lower().split()
                    matches = sum(1 for word in concept_words if word in text_lower)
                    
                    if matches > 0:
                        # Calculate confidence based on word matches
                        confidence = min(0.9, 0.4 + (matches / len(concept_words)) * 0.5)
                        concepts_with_confidence.append((concept, confidence))
        
        return concepts_with_confidence

    def extract_concepts(self, question_text: str) -> List[str]:
        """
        Enhanced concept extraction with domain detection and confidence scoring.
        """
        if not question_text:
            return []
        
        # Detect domain
        domain = self._detect_subject_domain(question_text)
        
        # Extract concepts using different methods
        pattern_concepts = self._extract_using_patterns(question_text)
        kb_concepts = self._extract_using_knowledge_base(question_text, domain)
        
        # Combine and deduplicate concepts
        all_concepts = {}
        
        for concept, confidence in pattern_concepts + kb_concepts:
            if concept in all_concepts:
                # Take maximum confidence if concept appears multiple times
                all_concepts[concept] = max(all_concepts[concept], confidence)
            else:
                all_concepts[concept] = confidence
        
        # Sort by confidence and return top concepts
        sorted_concepts = sorted(all_concepts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 concepts with confidence > 0.6
        final_concepts = [concept for concept, conf in sorted_concepts[:5] if conf > 0.6]
        
        # If no high-confidence concepts, return top 3 anyway
        if not final_concepts and sorted_concepts:
            final_concepts = [concept for concept, _ in sorted_concepts[:3]]
        
        # If still no concepts, provide a fallback based on domain
        if not final_concepts:
            fallback_concepts = {
                'history': ['Historical Analysis'],
                'economics': ['Economic Theory'],
                'mathematics': ['Mathematical Concepts'],
                'physics': ['Physics Principles'],
                'general': ['Academic Knowledge']
            }
            final_concepts = fallback_concepts.get(domain, ['General Knowledge'])
        
        return final_concepts

    def extract_concepts_with_confidence(self, question_text: str) -> List[Dict[str, any]]:
        """
        Extract concepts with detailed confidence information.
        """
        if not question_text:
            return []
        
        domain = self._detect_subject_domain(question_text)
        pattern_concepts = self._extract_using_patterns(question_text)
        kb_concepts = self._extract_using_knowledge_base(question_text, domain)
        
        # Combine concepts with metadata
        concept_details = []
        all_concepts = {}
        
        for concept, confidence in pattern_concepts + kb_concepts:
            if concept in all_concepts:
                all_concepts[concept] = max(all_concepts[concept], confidence)
            else:
                all_concepts[concept] = confidence
        
        for concept, confidence in sorted(all_concepts.items(), key=lambda x: x[1], reverse=True)[:5]:
            concept_details.append({
                'concept': concept,
                'confidence': confidence,
                'domain': domain,
                'extraction_method': 'pattern' if any(c == concept for c, _ in pattern_concepts) else 'knowledge_base'
            })
        
        return concept_details


