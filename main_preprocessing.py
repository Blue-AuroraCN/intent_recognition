# Import various preprocessing modules
from preprocessing.data_loader import load_data
from preprocessing.basic_stats import analyze_basic_stats
from preprocessing.missing_values import analyze_missing_values

# Load data
output_dir='data_exploration'
data_file = 'data/original_data.jsonl'
data = load_data(data_file)

# 1. Basic statistical analysis
df, basic_stats = analyze_basic_stats(data, output_dir)

# 2. Missing values analysis
missing_values = analyze_missing_values(df, output_dir)
