# Concept Extraction from Competitive Exam Questions
Roll no. 23b2289
## Project Overview

This project provides a robust and adaptable solution for extracting underlying concepts from competitive exam questions. The primary objective is to analyze a given set of questions (e.g., from UPSC - Ancient History) and identify the core concepts being tested in each question (e.g., "Indus Valley Civilization", "Gupta Period Literature").

Crucially, this solution is designed to be **cost-effective** by avoiding direct reliance on expensive Large Language Model (LLM) APIs for the core extraction logic. However, it is built with a modular architecture that allows for seamless future integration with LLM APIs when desired.

This tool aims to assist in understanding the conceptual distribution of past questions, which can aid in curriculum mapping, study analytics, or content generation.

## Features

- **Hybrid Concept Extraction**: Combines the power of the Rapid Automatic Keyword Extraction (RAKE) algorithm with a customizable keyword dictionary for accurate and relevant concept identification.
- **Cost-Effective**: Core extraction logic operates without direct LLM API calls, making it suitable for large datasets and cost-sensitive environments.
- **Future-Proof LLM Integration**: Designed with an abstract interface and a simulated LLM component to facilitate easy integration with real LLM APIs (e.g., OpenAI, Anthropic) in the future.
- **Domain Adaptability**: Easily configurable with subject-specific custom dictionaries and stop words, allowing the solution to be applied across various domains (e.g., History, Economics, Mathematics, Physics).
- **CSV-based Input/Output**: Processes questions from and outputs concepts to standard CSV files.
- **Comprehensive Documentation**: Includes detailed explanations of the solution architecture, design choices, and usage.
- **Interactive Jupyter Notebook**: A solution.ipynb file provides an interactive environment to explore and run the code.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   The main.py script will automatically attempt to download the necessary NLTK data (punkt, punkt_tab, and stopwords) if they are not already present.

## Usage

### Project Structure

```
.
├── main.py                 # Entry point, handles CLI and user code
├── concept_extractor.py    # Core concept extraction logic (RAKE, pre-processing, dictionary)
├── csv_reader.py           # Reads CSV from resources/ and returns data
├── llm_interface.py        # Abstract interface for LLM integration
├── simulated_llm.py        # Simulated LLM implementation
├── resources/              # Folder containing subject CSVs
│   ├── ancient_history.csv
│   ├── economics.csv
│   ├── mathematics.csv
│   └── physics.csv
├── dictionaries/           # Folder for domain-specific keyword dictionaries
│   ├── ancient_history_concepts.csv
│   ├── economics_concepts.csv
│   ├── mathematics_concepts.csv
│   └── physics_concepts.csv
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation (this file)
└── solution.ipynb         # Interactive Jupyter Notebook
```

### Running the Concept Extractor

Use the main.py script with the `--subject` argument to specify the domain of the questions. The script will read the corresponding CSV from the resources/ folder and use the dictionary from the dictionaries/ folder.

```bash
python main.py --subject=<subject_name>
```

#### Example (Ancient History):
```bash
python main.py --subject=ancient_history
```

This will process `resources/ancient_history.csv` and output `output_concepts.csv` with extracted concepts.

#### Available Subjects:
- `ancient_history` - Ancient History questions
- `economics` - Economics questions  
- `mathematics` - Mathematics questions
- `physics` - Physics questions

### Using the Simulated LLM

To test the LLM integration layer without making actual API calls, use the `--use_llm` flag:

```bash
python main.py --subject=<subject_name> --use_llm
```

#### Example (Simulated LLM for Ancient History):
```bash
python main.py --subject=ancient_history --use_llm
```

### Interactive Exploration with Jupyter Notebook

The `solution.ipynb` file provides an interactive way to understand and run the code. It includes all the code, explanations, and demonstrations.

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `solution.ipynb` in your browser.
3. Run the cells sequentially to see the concept extraction in action.

## Solution Overview

The solution employs a **Hybrid Concept Extraction System**:

1. **Core Extraction (Non-LLM)**: Utilizes the RAKE (Rapid Automatic Keyword Extraction) algorithm for identifying key phrases and concepts. RAKE is chosen for its domain independence, simplicity, and effectiveness without requiring external API calls.

2. **Custom Keyword Dictionary**: Augments RAKE by allowing human-curated mappings of specific terms to predefined concepts (e.g., "Harappan" -> "Indus Valley Civilization"). This ensures high precision for known concepts.

3. **Modular Design for LLM Integration**: An abstract `LLMConceptExtractor` interface is defined, allowing for seamless integration of real LLM APIs in the future. A `SimulatedLLM` class is provided for immediate testing of this integration layer.

This approach balances cost-effectiveness with the ability to scale to more advanced LLM-based solutions.

## LLM Prompt Format for Testing

As per the problem statement, the suggested LLM prompt format for testing is:

```
"Given the question: [question_text], identify the historical concept(s) this question is based on."
```

This format is used by the `simulated_llm.py` and would be adopted by any future real LLM integration.

## Sample Output

When you run the concept extraction, you'll see output similar to:

```
Reading questions from resources\ancient_history.csv...
Successfully loaded 5 questions from resources\ancient_history.csv
Using Hybrid Concept Extractor (RAKE + Custom Dictionary).
Concept extraction complete. Results saved to output_concepts.csv

--- Sample Extracted Concepts ---
Question Number: 1
Question: Which of the following was a feature of the Harappan civilization?
Concepts: Indus Valley Civilization; Following; Harappan Civilization; Feature

--- Summary ---
Total questions processed: 5
Questions with extracted concepts: 5
```

## File Formats

### Question CSV Format
```csv
Question Number,Question,Option A,Option B,Option C,Option D,Answer
1,"Question text here","Option A text","Option B text","Option C text","Option D text",A
```

### Dictionary CSV Format
```csv
keyword,concept
harappan,Indus Valley Civilization
mauryan,Mauryan Empire
```

### Output CSV Format
```csv
Question Number,Question,Concepts
1,"Question text here","Concept1; Concept2; Concept3"
```

## Customization

### Adding New Subjects

1. Create a new questions CSV file in the `resources/` folder
2. Create a corresponding concepts dictionary in the `dictionaries/` folder
3. Run the extraction: `python main.py --subject=your_new_subject`

### Modifying Extraction Logic

- **RAKE parameters**: Modify the `ConceptExtractor` class in `concept_extractor.py`
- **Custom dictionaries**: Edit the CSV files in the `dictionaries/` folder
- **LLM integration**: Implement the `LLMConceptExtractor` interface in `llm_interface.py`

## Technical Details

### Dependencies
- **pandas**: Data manipulation and CSV handling
- **nltk**: Natural language processing and tokenization
- **rake-nltk**: RAKE algorithm implementation

### Architecture
The project follows a modular architecture with clear separation of concerns:
- **Data Layer**: `csv_reader.py` handles all file I/O
- **Processing Layer**: `concept_extractor.py` contains the core logic
- **Interface Layer**: `llm_interface.py` defines the contract for LLM integration
- **Application Layer**: `main.py` provides the CLI interface

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**: The script automatically downloads required NLTK data. If issues persist, manually run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   ```

2. **CSV Parsing Errors**: Ensure CSV files have proper formatting with quotes around text containing commas.

3. **Missing Subject Files**: Verify that both the questions CSV and dictionary CSV exist for your subject.

## Future Enhancements

- Integration with real LLM APIs (OpenAI, Anthropic, etc.)
- Advanced preprocessing techniques
- Machine learning-based concept classification
- Web interface for easier interaction
- Batch processing capabilities
- Performance optimizations for large datasets

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project.
