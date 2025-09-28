
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
import os

def evaluate_model(model, label_encoder, X_test, y_test):
    """
    Evaluate model performance

    Args:
        model: Trained model
        label_encoder: Label encoder
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels

    Returns:
        dict: Evaluation results
    """
    print("Evaluating model performance...")

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Get class names
    class_names = label_encoder.classes_

    # Calculate classification report
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names,
        output_dict=True
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names.tolist()
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return results

def plot_confusion_matrix(results, output_dir=None, figsize=(10, 8)):
    """
    Plot confusion matrix

    Args:
        results (dict): Evaluation results
        output_dir (str): Output directory, if provided save the image
        figsize (tuple): Figure size
    """
    if not results:
        raise ValueError("No evaluation results available.")

    cm = np.array(results['confusion_matrix'])
    class_names = results['class_names']

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")

    plt.show()

def plot_classification_report(results, output_dir=None, figsize=(12, 8)):
    """
    Plot classification report heatmap

    Args:
        results (dict): Evaluation results
        output_dir (str): Output directory, if provided save the image
        figsize (tuple): Figure size
    """
    if not results:
        raise ValueError("No evaluation results available.")

    report = results['classification_report']
    class_names = results['class_names']

    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {metric: [] for metric in metrics}

    for class_name in class_names:
        for metric in metrics:
            class_metrics[metric].append(report[class_name][metric])

    # Create dataframe
    import pandas as pd
    df = pd.DataFrame(class_metrics, index=class_names)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Classification Report')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'classification_report.png')
        plt.savefig(output_path)
        print(f"Classification report saved to {output_path}")

    plt.show()

def save_results(results, output_dir, filename='evaluation_results.json'):
    """
    Save evaluation results

    Args:
        results (dict): Evaluation results
        output_dir (str): Output directory
        filename (str): Output filename
    """
    if not results:
        raise ValueError("No evaluation results available.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_path}")
