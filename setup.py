"""
Setup script for the Concept Extraction project.
"""
import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk_data = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords')
        ]
        
        for data_path, package_name in nltk_data:
            try:
                nltk.data.find(data_path)
                print(f"✓ {package_name} already downloaded")
            except LookupError:
                print(f"Downloading {package_name}...")
                nltk.download(package_name)
                print(f"✓ {package_name} downloaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    directories = [
        "resources",
        "dictionaries", 
        "output",
        "batch_output",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def validate_installation():
    """Validate that the installation was successful."""
    print("Validating installation...")
    
    try:
        # Test imports
        import pandas as pd
        import nltk
        from rake_nltk import Rake
        print("✓ Core dependencies imported successfully")
        
        # Test NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK data available")
        except LookupError:
            print("✗ NLTK data not found")
            return False
        
        # Test our modules
        from csv_reader import read_questions_csv
        from concept_extractor import ConceptExtractor
        from simulated_llm import SimulatedLLM
        print("✓ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def run_demo():
    """Run a quick demo to test the system."""
    print("Running demo...")
    
    try:
        # Check if we have sample data
        if os.path.exists("resources/ancient_history.csv"):
            print("Running concept extraction demo...")
            from main import main
            import sys
            
            # Temporarily modify sys.argv for demo
            original_argv = sys.argv
            sys.argv = ["main.py", "--subject", "ancient_history"]
            
            try:
                main()
                print("✓ Demo completed successfully")
            finally:
                sys.argv = original_argv
        else:
            print("⚠ No sample data found. Skipping demo.")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 50)
    print("Concept Extraction Project Setup")
    print("=" * 50)
    
    success = True
    
    # Step 1: Install requirements
    if not install_requirements():
        success = False
    
    # Step 2: Download NLTK data
    if success and not download_nltk_data():
        success = False
    
    # Step 3: Create directories
    if success and not create_directories():
        success = False
    
    # Step 4: Validate installation
    if success and not validate_installation():
        success = False
    
    # Step 5: Run demo (optional)
    if success:
        run_demo()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Setup completed successfully!")
        print("\nYou can now run:")
        print("  python main.py --subject ancient_history")
        print("  python batch_processor.py --subjects ancient_history economics")
        print("  jupyter notebook solution.ipynb")
    else:
        print("✗ Setup encountered errors. Please check the output above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
