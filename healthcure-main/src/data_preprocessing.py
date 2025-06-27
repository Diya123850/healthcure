import pandas as pd

def load_and_clean_data(filepath):
    """
    Loads data from a CSV file, drops missing values, and encodes categorical variables.
    
    Args:
        filepath (str): Path to the CSV data file.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    data = pd.read_csv(filepath)
    # Drop missing values
    data = data.dropna()
    # One-hot encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    return data

if __name__ == "__main__":
    # Example usage
    input_path = "data/sample_patient_data.csv"
    output_path = "data/preprocessed_patient_data.csv"
    data = load_and_clean_data(input_path)
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
