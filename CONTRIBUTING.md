# Contributing to Concept Extraction Project

Thank you for considering contributing to the Concept Extraction project! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** to provide all necessary information
3. **Include steps to reproduce** the issue
4. **Provide sample data** if relevant

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Ensure tests pass** (`python tests.py`)
6. **Update documentation** if needed
7. **Commit your changes** (`git commit -m 'Add amazing feature'`)
8. **Push to the branch** (`git push origin feature/amazing-feature`)
9. **Open a Pull Request**

## Coding Standards

### Python Code Style
- Follow **PEP 8** style guidelines
- Use **type hints** where possible
- Write **comprehensive docstrings**
- Keep functions **focused and small**
- Use **meaningful variable names**

### Documentation
- Update **README.md** for major changes
- Add **inline comments** for complex logic
- Update **docstrings** for all functions
- Include **examples** in documentation

### Testing
- Write **unit tests** for new functions
- Ensure **test coverage** is maintained
- Test with **different data formats**
- Include **edge cases** in tests

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/concept-extraction.git
   cd concept-extraction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup**:
   ```bash
   python setup.py
   ```

4. **Run tests**:
   ```bash
   python tests.py
   ```

## Project Structure

```
concept-extraction/
├── main.py                    # Main entry point
├── concept_extractor.py       # Core extraction logic
├── simulated_llm.py           # LLM simulation
├── csv_reader.py              # Data reading utilities
├── config_manager.py          # Configuration management
├── batch_processor.py         # Batch processing
├── evaluation_utils.py        # Performance evaluation
├── tests.py                   # Test suite
├── setup.py                   # Setup script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # This file
├── LICENSE                    # License information
├── .gitignore                 # Git ignore patterns
├── resources/                 # Sample question data
├── dictionaries/              # Concept dictionaries
└── batch_output/              # Output directory
```

## Adding New Features

### New Extraction Methods
1. Implement the `LLMConceptExtractor` interface
2. Add configuration options in `config_manager.py`
3. Update the factory pattern in `enhanced_llm_interface.py`
4. Add comprehensive tests
5. Update documentation

### New Subject Domains
1. Add CSV files to `resources/` directory
2. Create concept dictionaries in `dictionaries/`
3. Test with existing tools
4. Document domain-specific considerations

### New Output Formats
1. Extend the output format options in `main.py`
2. Update the concept extractor export methods
3. Add validation for new formats
4. Include examples in documentation

## Performance Guidelines

- **Profile your changes** for performance impact
- **Test with large datasets** when possible
- **Optimize for memory usage** in batch processing
- **Consider scalability** for production use

## Documentation Guidelines

### Code Documentation
```python
def extract_concepts(self, question_text: str) -> List[str]:
    """
    Extract concepts from a single question using hybrid approach.
    
    Args:
        question_text (str): The input question text to analyze
        
    Returns:
        List[str]: List of extracted concept names
        
    Raises:
        ValueError: If question_text is None or empty
        
    Example:
        >>> extractor = ConceptExtractor()
        >>> concepts = extractor.extract_concepts("What is GDP?")
        >>> print(concepts)
        ['Economic Indicators', 'National Income']
    """
```

### README Updates
- Keep the **getting started** section up to date
- Include **new examples** for major features
- Update **command-line options** documentation
- Maintain **troubleshooting** section

## Release Process

1. **Update version** numbers
2. **Update CHANGELOG**
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Publish release notes**

## Community Guidelines

- **Be respectful** and inclusive
- **Help newcomers** get started
- **Share knowledge** and best practices
- **Provide constructive feedback**

## Questions?

Feel free to:
- **Open an issue** for questions
- **Start a discussion** for ideas
- **Contact maintainers** directly

Thank you for contributing! 🎉
