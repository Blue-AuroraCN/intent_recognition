# Import various preprocessing modules
from preprocessing.data_loader import load_data
from preprocessing.basic_stats import analyze_basic_stats
from preprocessing.missing_values import analyze_missing_values
from preprocessing.duplicate_values import analyze_duplicate_values
from preprocessing.topic_analysis import analyze_topic_distribution
from preprocessing.text_length_analysis import analyze_text_length
from preprocessing.background_analysis import analyze_background_info
from preprocessing.text_content_analysis import analyze_text_content
from preprocessing.data_cleaning import clean_and_save_data, save_statistics
from preprocessing.type_converter import convert_numpy_to_python

# Load data
output_dir='data_exploration'
data_file = 'data/original_data.jsonl'
data = load_data(data_file)

# 1. Basic statistical analysis
df, basic_stats = analyze_basic_stats(data, output_dir)

# 2. Missing values analysis
missing_values = analyze_missing_values(df, output_dir)

# 3. Duplicate values analysis
duplicate_monologues, duplicate_rows = analyze_duplicate_values(df)

# 4. Topic distribution analysis
topic_counts = analyze_topic_distribution(df, output_dir)

# 5. Text length analysis
text_length_stats = analyze_text_length(df, output_dir)

# 6. Background information analysis
background_stats = analyze_background_info(df, output_dir)

# 7. Text content analysis
most_common_words, topic_word_stats = analyze_text_content(df, output_dir)

# 8. Clean and save data
df_cleaned = clean_and_save_data(df, output_dir)

# 9. Summarize and save all statistics
stats = {
    'original_size': len(df),
    'cleaned_size': len(df_cleaned),
    'duplicate_rows': duplicate_rows,
    'duplicate_monologues': duplicate_monologues,
    'missing_values': missing_values,
    'topic_counts': topic_counts.to_dict(),
    'text_length_stats': text_length_stats.to_dict(),
    'most_common_words': most_common_words,
    'background_stats': background_stats,
    'topic_word_stats': topic_word_stats
}

# Convert NumPy data types to Python native types
stats = convert_numpy_to_python(stats)

save_statistics(stats, output_dir)

