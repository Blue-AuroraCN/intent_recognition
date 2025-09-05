
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_background_info(df, output_dir='data_exploration'):
    """
    Analyze background information

    Parameters:
        df (pd.DataFrame): Data DataFrame
        output_dir (str): Output directory

    Returns:
        dict: Background information statistics
    """
    print("=== Background Information Analysis ===")
    # Extract background information
    background_df = pd.json_normalize(df['background'])

    # Analyze distribution of each background feature
    background_stats = {}
    os.makedirs(output_dir, exist_ok=True)

    for col in background_df.columns:
        print(f"{col} distribution:")
        value_counts = background_df[col].value_counts()
        print(value_counts)

        # Save statistics
        background_stats[col] = value_counts.to_dict()

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'), dpi=300)
        plt.close()

    return background_stats
