import pandas as pd

def read_questions_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file containing competitive exam questions into a pandas DataFrame.

    Args:
        file_path (str): The absolute path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the questions, or an empty DataFrame if an error occurs.
    """
    try:
        # Try reading with different parameters to handle parsing issues
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Validate expected columns
        expected_columns = ['Question Number', 'Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Answer']
        if not all(col in df.columns for col in expected_columns):
            print(f"Error: CSV file {file_path} is missing one or more expected columns.")
            print(f"Expected: {expected_columns}")
            print(f"Found: {list(df.columns)}")
            return pd.DataFrame(columns=expected_columns)
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        print(f"Successfully loaded {len(df)} questions from {file_path}")
        return df
        
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file {file_path}: {e}")
        print("Trying with different parsing options...")
        try:
            # Try with error_bad_lines=False for older pandas versions
            # or on_bad_lines='skip' for newer versions
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            print(f"Successfully loaded {len(df)} questions with some lines skipped")
            return df
        except Exception as e2:
            print(f"Failed to read CSV even with error recovery: {e2}")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return pd.DataFrame()


