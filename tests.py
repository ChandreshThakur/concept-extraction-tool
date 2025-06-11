"""
Comprehensive test suite for the Concept Extraction project.
"""
import unittest
import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_reader import read_questions_csv
from concept_extractor import ConceptExtractor
from simulated_llm import SimulatedLLM
from config_manager import ConfigManager, ProjectConfig

class TestCSVReader(unittest.TestCase):
    """Test cases for CSV reading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, "test_questions.csv")
        
        # Create test CSV
        test_data = """Question Number,Question,Option A,Option B,Option C,Option D,Answer
1,"Test question 1","Option A","Option B","Option C","Option D",A
2,"Test question 2","Option A","Option B","Option C","Option D",B"""
        
        with open(self.test_csv, 'w', encoding='utf-8') as f:
            f.write(test_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_read_valid_csv(self):
        """Test reading a valid CSV file."""
        df = read_questions_csv(self.test_csv)
        self.assertEqual(len(df), 2)
        self.assertIn('Question', df.columns)
        self.assertEqual(df.iloc[0]['Question Number'], 1)
    
    def test_read_nonexistent_csv(self):
        """Test reading a non-existent CSV file."""
        df = read_questions_csv("nonexistent.csv")
        self.assertTrue(df.empty)
    
    def test_read_invalid_csv(self):
        """Test reading an invalid CSV file."""
        invalid_csv = os.path.join(self.test_dir, "invalid.csv")
        with open(invalid_csv, 'w') as f:
            f.write("invalid,csv,format\n1,2")  # Missing column
        
        df = read_questions_csv(invalid_csv)
        # Should handle gracefully
        self.assertIsInstance(df, pd.DataFrame)

class TestConceptExtractor(unittest.TestCase):
    """Test cases for concept extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test dictionary
        self.dict_file = os.path.join(self.test_dir, "test_concepts.csv")
        dict_data = """keyword,concept
test,Test Concept
sample,Sample Concept
example,Example Concept"""
        
        with open(self.dict_file, 'w', encoding='utf-8') as f:
            f.write(dict_data)
        
        self.extractor = ConceptExtractor(custom_dict_file=self.dict_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_extract_concepts_with_dictionary_match(self):
        """Test concept extraction with dictionary matches."""
        question = "This is a test question with sample data."
        concepts = self.extractor.extract_concepts(question)
        
        # Should find concepts from dictionary
        concept_names = [c for c in concepts]
        self.assertIn('Test Concept', concept_names)
        self.assertIn('Sample Concept', concept_names)
    
    def test_extract_concepts_empty_question(self):
        """Test concept extraction with empty question."""
        concepts = self.extractor.extract_concepts("")
        self.assertEqual(concepts, [])
        
        concepts = self.extractor.extract_concepts(None)
        self.assertEqual(concepts, [])
    
    def test_extract_concepts_no_matches(self):
        """Test concept extraction with no dictionary matches."""
        question = "This question has no matching keywords."
        concepts = self.extractor.extract_concepts(question)
        
        # Should still return some concepts from RAKE or fallback
        self.assertIsInstance(concepts, list)
    
    def test_extract_from_dataframe(self):
        """Test extracting concepts from a DataFrame."""
        df = pd.DataFrame({
            'Question Number': [1, 2],
            'Question': ['This is a test question.', 'Another sample question.'],
            'Option A': ['A', 'A'],
            'Option B': ['B', 'B'],
            'Option C': ['C', 'C'],
            'Option D': ['D', 'D'],
            'Answer': ['A', 'B']
        })
        
        result_df = self.extractor.extract_concepts_from_dataframe(df)
        
        self.assertIn('Concepts', result_df.columns)
        self.assertEqual(len(result_df), 2)
        
        # Check that concepts were extracted
        for concepts in result_df['Concepts']:
            self.assertIsInstance(concepts, str)

class TestSimulatedLLM(unittest.TestCase):
    """Test cases for simulated LLM functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = SimulatedLLM()
    
    def test_extract_historical_concepts(self):
        """Test extraction of historical concepts."""
        question = "What was the main feature of the Harappan civilization?"
        concepts = self.llm.extract_concepts(question)
        
        self.assertIsInstance(concepts, list)
        self.assertTrue(len(concepts) > 0)
        
        # Should detect historical concepts
        concept_str = ' '.join(concepts).lower()
        self.assertTrue(any(term in concept_str for term in ['indus', 'civilization', 'history']))
    
    def test_extract_economic_concepts(self):
        """Test extraction of economic concepts."""
        question = "What is the effect of monetary policy on inflation?"
        concepts = self.llm.extract_concepts(question)
        
        self.assertIsInstance(concepts, list)
        concept_str = ' '.join(concepts).lower()
        self.assertTrue(any(term in concept_str for term in ['monetary', 'economic', 'policy']))
    
    def test_extract_empty_question(self):
        """Test extraction with empty question."""
        concepts = self.llm.extract_concepts("")
        self.assertEqual(concepts, [])
    
    def test_extract_concepts_with_confidence(self):
        """Test extraction with confidence scores."""
        question = "What was the main feature of the Harappan civilization?"
        concepts_with_conf = self.llm.extract_concepts_with_confidence(question)
        
        self.assertIsInstance(concepts_with_conf, list)
        if concepts_with_conf:
            for item in concepts_with_conf:
                self.assertIn('concept', item)
                self.assertIn('confidence', item)
                self.assertIn('domain', item)

class TestConfigManager(unittest.TestCase):
    """Test cases for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        manager = ConfigManager(self.config_file)
        config = manager.get_config()
        
        self.assertIsInstance(config, ProjectConfig)
        self.assertEqual(config.extraction.confidence_threshold, 0.5)
        self.assertEqual(config.llm.provider, "simulated")
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        manager = ConfigManager(self.config_file)
        
        # Modify configuration
        manager.update_config(
            extraction={'confidence_threshold': 0.7, 'max_concepts': 15}
        )
        
        # Save configuration
        manager.save_config()
        
        # Load in new manager
        new_manager = ConfigManager(self.config_file)
        new_config = new_manager.get_config()
        
        self.assertEqual(new_config.extraction.confidence_threshold, 0.7)
        self.assertEqual(new_config.extraction.max_concepts, 15)
    
    def test_validate_config(self):
        """Test configuration validation."""
        manager = ConfigManager(self.config_file)
        
        # Valid configuration
        issues = manager.validate_config()
        # Note: May have issues if directories don't exist, which is expected
        
        # Invalid configuration
        manager.update_config(
            extraction={'confidence_threshold': 1.5}  # Invalid value
        )
        issues = manager.validate_config()
        self.assertTrue(any('confidence_threshold' in issue for issue in issues))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test resources
        self.resources_dir = os.path.join(self.test_dir, "resources")
        self.dict_dir = os.path.join(self.test_dir, "dictionaries")
        
        os.makedirs(self.resources_dir)
        os.makedirs(self.dict_dir)
        
        # Create test CSV
        test_csv = os.path.join(self.resources_dir, "test_subject.csv")
        csv_data = """Question Number,Question,Option A,Option B,Option C,Option D,Answer
1,"What is the significance of Harappan civilization?","Urban planning","Agriculture","Trade","All of above",D
2,"Which economic policy affects inflation?","Monetary policy","Trade policy","Both","None",A"""
        
        with open(test_csv, 'w', encoding='utf-8') as f:
            f.write(csv_data)
        
        # Create test dictionary
        test_dict = os.path.join(self.dict_dir, "test_subject_concepts.csv")
        dict_data = """keyword,concept
harappan,Indus Valley Civilization
monetary policy,Monetary Economics
inflation,Economic Indicators"""
        
        with open(test_dict, 'w', encoding='utf-8') as f:
            f.write(dict_data)
        
        # Change to test directory
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_hybrid_extraction(self):
        """Test complete hybrid extraction workflow."""
        # Read questions
        questions_df = read_questions_csv("resources/test_subject.csv")
        self.assertFalse(questions_df.empty)
        
        # Extract concepts
        extractor = ConceptExtractor(custom_dict_file="dictionaries/test_subject_concepts.csv")
        result_df = extractor.extract_concepts_from_dataframe(questions_df)
        
        # Verify results
        self.assertIn('Concepts', result_df.columns)
        self.assertTrue(all(len(concepts) > 0 for concepts in result_df['Concepts']))
        
        # Check for expected concepts
        all_concepts = ' '.join(result_df['Concepts'])
        self.assertIn('Indus Valley Civilization', all_concepts)
        self.assertIn('Monetary Economics', all_concepts)
    
    def test_end_to_end_llm_extraction(self):
        """Test complete LLM extraction workflow."""
        # Read questions
        questions_df = read_questions_csv("resources/test_subject.csv")
        self.assertFalse(questions_df.empty)
        
        # Extract concepts using simulated LLM
        llm = SimulatedLLM()
        questions_df["Concepts"] = questions_df["Question"].apply(llm.extract_concepts)
        questions_df["Concepts"] = questions_df["Concepts"].apply(
            lambda concepts: "; ".join(concepts) if concepts else ""
        )
        
        # Verify results
        self.assertTrue(all(len(concepts) > 0 for concepts in questions_df['Concepts']))

def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCSVReader,
        TestConceptExtractor,
        TestSimulatedLLM,
        TestConfigManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
