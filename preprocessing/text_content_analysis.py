
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def analyze_text_content(df, output_dir='data_exploration'):
    """
    Analyze text content

    Parameters:
        df (pd.DataFrame): Data DataFrame
        output_dir (str): Output directory

    Returns:
        tuple: (list of most common words, word frequency statistics grouped by topic)
    """
    print("=== Text Content Analysis ===")
    # Combine all text
    all_text = ' '.join(df['monologue'].values)

    # Clean text and tokenize
    words = re.findall(r'\w+', all_text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Word frequency statistics
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(20)
    print("Top 20 most common words:")
    print(most_common_words)

    # Visualize word frequency
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common_words)
    sns.barplot(x=list(words), y=list(counts))
    plt.title('Top 20 Most Common Words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_common_words.png'), dpi=300)
    plt.close()

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wordcloud.png'), dpi=300)
    plt.close()

    # Analyze text content by topic
    print("=== Text Content Analysis by Topic ===")
    topic_word_stats = {}

    for topic in df['topic'].unique():
        topic_text = ' '.join(df[df['topic'] == topic]['monologue'].values)
        topic_words = re.findall(r'\w+', topic_text.lower())
        
        # Remove stopwords
        topic_words = [word for word in topic_words if word not in stop_words]
        topic_word_counts = Counter(topic_words)
        topic_most_common = topic_word_counts.most_common(10)

        print(f"Topic: {topic}")
        print("Top 10 most common words:")
        print(topic_most_common)

        # Save statistics
        topic_word_stats[topic] = topic_most_common

        # Visualization
        plt.figure(figsize=(10, 6))
        words, counts = zip(*topic_most_common)
        sns.barplot(x=list(words), y=list(counts))
        plt.title(f'Topic: {topic} - Top 10 Most Common Words')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topic_{topic}_common_words.png'), dpi=300)
        plt.close()

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(topic_words))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic: {topic} - Word Cloud')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topic_{topic}_wordcloud.png'), dpi=300)
        plt.close()

    return most_common_words, topic_word_stats
