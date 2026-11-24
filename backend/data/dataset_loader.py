import pandas as pd

def load_dataset(file_path, target_column):
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=["id"])
    
    if df.empty:
        raise ValueError("The dataset is empty.")
    
    if target_column not in df.columns:
        raise ValueError(f"The target column '{target_column}' is not present in the dataset.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y
