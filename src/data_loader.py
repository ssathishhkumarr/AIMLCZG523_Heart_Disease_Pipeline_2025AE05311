import pandas as pd
import os

def download_and_clean_data(output_path="data/heart_disease.csv"):
    print("Downloading dataset from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Define the 14 feature columns[cite: 1]
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    # Read the CSV, treating '?' as missing values (NaN)
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # Clean data: Impute missing values with the median of their respective columns[cite: 1]
    df.fillna(df.median(), inplace=True)
    
    # Preprocess: Convert multi-class target to a binary target (0 = no disease, 1 = disease)[cite: 1]
    df["target"] = (df["target"] > 0).astype(int)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    download_and_clean_data()