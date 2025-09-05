
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_text_length(df, output_dir='data_exploration'):
    """
    Analyze monologue text length

    Parameters:
        df (pd.DataFrame): Data DataFrame
        output_dir (str): Output directory

    Returns:
        pd.Series: Text length statistics
    """
    print("=== Monologue Text Length Analysis ===")
    df['text_length'] = df['monologue'].apply(len)
    print(f"Text length statistics:")
    print(df['text_length'].describe())

    # Visualize text length distribution
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title('Monologue Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_length_distribution.png'), dpi=300)
    plt.close()

    # Box plot of text length grouped by topic
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='topic', y='text_length', data=df)
    plt.title('Text Length Distribution by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Text Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_length_by_topic.png'), dpi=300)
    plt.close()

    return df['text_length'].describe()
