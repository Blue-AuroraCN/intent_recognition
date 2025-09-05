
import pandas as pd
import json
import os

def clean_and_save_data(df, output_dir='data_exploration'):
    """
    Clean duplicate data and save results

    Parameters:
        df (pd.DataFrame): Original data DataFrame
        output_dir (str): Output directory

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print("=== Saving Processed Data ===")

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Handle missing values in monologue field (including empty strings or zero-length cases)
    if 'monologue' in df_copy.columns:
        # Treat empty string or zero-length monologue as missing values
        df_copy.loc[df_copy['monologue'].astype(str).str.strip() == '', 'monologue'] = None

        # Remove rows with missing monologue values
        before_drop = len(df_copy)
        df_copy = df_copy.dropna(subset=['monologue'])
        after_drop = len(df_copy)
        print(f"Removed {before_drop - after_drop} rows with missing monologue values")
        print(f"Dataset size after removing missing monologue values: {after_drop}")

    # Process duplicates based only on monologue field
    before_dedup = len(df_copy)
    df_cleaned = df_copy.drop_duplicates(subset=['monologue'])
    after_dedup = len(df_cleaned)
    print(f"Removed {before_dedup - after_dedup} duplicate rows based on monologue field")
    print(f"Dataset size after removing duplicates: {after_dedup}")

    # Save cleaned data
    os.makedirs(output_dir, exist_ok=True)
    df_cleaned.to_json(os.path.join(output_dir, 'cleaned_data.jsonl'), orient='records', lines=True)

    return df_cleaned

def save_statistics(stats, output_dir='data_exploration'):
    """
    Save statistics

    Parameters:
        stats (dict): Statistics dictionary
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"Data exploration completed! Results saved in {output_dir} directory")


