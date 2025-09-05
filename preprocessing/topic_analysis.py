
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_topic_distribution(df, output_dir='data_exploration'):
    """
    Analyze topic distribution

    Parameters:
        df (pd.DataFrame): Data DataFrame
        output_dir (str): Output directory

    Returns:
        pd.Series: Topic counts
    """
    print("=== Topic Distribution Analysis ===")
    topic_counts = df['topic'].value_counts()
    print(topic_counts)

    # Visualize topic distribution
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=topic_counts.index, y=topic_counts.values)
    plt.title('Topic Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_distribution.png'), dpi=300)
    plt.close()

    # Pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%')
    plt.title('Topic Proportion')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_pie.png'), dpi=300)
    plt.close()

    return topic_counts
