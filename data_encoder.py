
import json
import numpy as np
from transformers import AutoModel
import torch
from tqdm import tqdm
import os

def load_model(model_id="jinaai/jina-embeddings-v2-small-en"):
    """
    Load model and preprocessor

    Args:
        model_id (str): Model ID

    Returns:
        tuple: (model, preprocessor, device)
    """
    print(f"Loading model {model_id}...")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model, device

def encode_texts(model, device, texts, batch_size=32):
    """
    Encode a list of texts

    Args:
        model: Loaded model
        device: Computing device
        texts (list): List of texts
        batch_size (int): Batch size

    Returns:
        np.ndarray: Text encoding results
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]

        # Get encodings
        batch_embeddings = model.encode(batch_texts, device=device)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def encode_data_from_jsonl(input_file, output_file, text_field="monologue", 
                          model_id="jinaai/jina-embeddings-v2-small-en", batch_size=32):
    """
    Read data from JSONL file, encode texts and save results

    Args:
        input_file (str): Input JSONL file path
        output_file (str): Output file path
        text_field (str): Text field name
        model_id (str): Model ID
        batch_size (int): Batch size
    """
    print(f"Loading data from {input_file}...")

    texts = []
    labels = []
    metadata = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading data"):
            data = json.loads(line.strip())
            texts.append(data[text_field])
            labels.append(data["topic"])
            metadata.append({k: v for k, v in data.items() if k not in [text_field, "topic"]})

    print(f"Loaded {len(texts)} samples")

    # Load model
    model, device = load_model(model_id)

    # Encode texts
    embeddings = encode_texts(model, device, texts, batch_size)

    # Save encoding results
    print(f"Saving encoded data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            result = {
                "embedding": embeddings[i].tolist(),
                "topic": labels[i],
                "metadata": metadata[i]
            }
            f.write(json.dumps(result) + '\n')

    print("Encoding completed successfully")


if __name__ == "__main__":
    encode_data_from_jsonl('data_exploration/cleaned_data.jsonl', 'outputs/encoded_data.jsonl')
