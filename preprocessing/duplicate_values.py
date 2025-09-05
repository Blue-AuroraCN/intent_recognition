
import pandas as pd

def analyze_duplicate_values(df):
    """
    Analyze duplicate values in the data

    Parameters:
        df (pd.DataFrame): Data DataFrame

    Returns:
        tuple: (number of duplicate monologues, number of duplicate rows)
    """
    print("=== Duplicate Values Check ===")
    # Detect duplicates based only on monologue field
    duplicate_monologues = df['monologue'].duplicated().sum()
    print(f"Number of duplicate monologues: {duplicate_monologues}")

    # Find the number of complete rows corresponding to duplicate monologues
    duplicate_rows = df.duplicated(subset=['monologue']).sum()
    print(f"Number of duplicate rows based on monologue: {duplicate_rows}")

    return duplicate_monologues, duplicate_rows
