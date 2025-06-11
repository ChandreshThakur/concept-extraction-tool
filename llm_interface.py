from abc import ABC, abstractmethod
from typing import List

class LLMConceptExtractor(ABC):
    """
    Abstract base class for LLM-based concept extraction.
    Defines the interface that all LLM concept extractors must implement.
    """

    @abstractmethod
    def extract_concepts(self, question_text: str) -> List[str]:
        """
        Extracts concepts from a given question text using an LLM.

        Args:
            question_text (str): The text of the competitive exam question.

        Returns:
            List[str]: A list of extracted concepts.
        """
        pass


