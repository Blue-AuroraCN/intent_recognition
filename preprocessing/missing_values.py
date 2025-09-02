
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_missing_values(df, output_dir='data_exploration'):
    """
    Analyze missing values in the data

    Parameters:
        df (pd.DataFrame): Data DataFrame
        output_dir (str): Output directory

    Returns:
        dict: Missing values statistics
    """
    print("=== Missing Values Check ===")

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Check if monologue field is empty string or has length 0
    if 'monologue' in df_copy.columns:
        # Treat empty string or zero-length monologue as missing values
        df_copy.loc[df_copy['monologue'].astype(str).str.strip() == '', 'monologue'] = None

    # Calculate missing values
    missing_values = df_copy.isnull().sum()
    print(missing_values)

    # Visualize missing values
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_copy.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'), dpi=300)
    plt.close()

    return missing_values.to_dict()


