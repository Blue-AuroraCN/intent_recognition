
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from tqdm import tqdm

def load_encoded_data(data_file):
    """
    Load encoded data

    Args:
        data_file (str): Path to the encoded data file

    Returns:
        tuple: (features, labels, metadata)
    """
    print(f"Loading encoded data from {data_file}...")

    embeddings = []
    labels = []
    metadata = []

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data = json.loads(line.strip())
            embeddings.append(data["embedding"])
            labels.append(data["topic"])
            metadata.append(data["metadata"])

    print(f"Loaded {len(embeddings)} samples")
    return np.array(embeddings), np.array(labels), metadata

def prepare_data(embeddings, labels, test_size=0.2, random_state=42):
    """
    Prepare training and testing data

    Args:
        embeddings (np.ndarray): Text embeddings
        labels (np.ndarray): Labels
        test_size (float): Proportion of the dataset for testing
        random_state (int): Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Split training and testing sets
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=test_size, 
        random_state=random_state, stratify=y_encoded
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    """
    Train logistic regression model

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        C (float): Inverse of regularization strength
        max_iter (int): Maximum number of iterations
        random_state (int): Random seed

    Returns:
        LogisticRegression: Trained model
    """
    print("Training logistic regression model...")

    # Create and train model
    model = LogisticRegression()

    model.fit(X_train, y_train)
    print("Model training completed")

    return model

def save_model(model, label_encoder, model_dir, model_name="intent_model.pkl", encoder_name="label_encoder.pkl"):
    """
    Save trained model and label encoder

    Args:
        model: Trained model
        label_encoder: Label encoder
        model_dir (str): Directory to save the model
        model_name (str): Model file name
        encoder_name (str): Encoder file name
    """
    if model is None:
        raise ValueError("Model has not been trained yet")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_name)
    encoder_path = os.path.join(model_dir, encoder_name)

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    print(f"Saving label encoder to {encoder_path}...")
    joblib.dump(label_encoder, encoder_path)

    print("Model and encoder saved successfully")

def load_model(model_dir, model_name="intent_model.pkl", encoder_name="label_encoder.pkl"):
    """
    Load trained model and label encoder

    Args:
        model_dir (str): Model directory
        model_name (str): Model file name
        encoder_name (str): Encoder file name

    Returns:
        tuple: (model, label_encoder)
    """
    model_path = os.path.join(model_dir, model_name)
    encoder_path = os.path.join(model_dir, encoder_name)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading label encoder from {encoder_path}...")
    label_encoder = joblib.load(encoder_path)

    print("Model and encoder loaded successfully")
    return model, label_encoder
