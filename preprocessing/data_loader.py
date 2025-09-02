
import json

def load_data(file_path):
    """
    Load data from JSONL file

    Parameters:
        file_path (str): Path to JSONL file

    Returns:
        list: List containing all data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
