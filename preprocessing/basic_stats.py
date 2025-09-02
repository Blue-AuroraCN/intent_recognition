
import pandas as pd
import os

def analyze_basic_stats(data, output_dir='data_exploration'):
    """
    Analyze basic statistics of the data

    Parameters:
        data (list): Original data list
        output_dir (str): Output directory

    Returns:
        tuple: (DataFrame, basic statistics dictionary)
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Basic statistics
    print("=== Basic Statistics ===")
    print(f"Dataset size: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    print("Data preview:")
    print(df.head())

    # Save basic statistics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'basic_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Dataset size: {len(df)}")
        f.write(f"Column names: {list(df.columns)}")
        f.write("Data preview:")
        f.write(df.head().to_string())

    # Statistics dictionary
    stats = {
        'dataset_size': len(df),
        'columns': list(df.columns)
    }

    return df, stats
