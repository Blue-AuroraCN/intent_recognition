
import os

from data_encoder import encode_data_from_jsonl
from model_training import (
    load_encoded_data, prepare_data, train_model, save_model
)
from model_evaluation import (
    evaluate_model, save_results, plot_confusion_matrix, plot_classification_report
)

# Define parameters directly
data_file = 'data_exploration/cleaned_data.jsonl'
encoded_data_file = 'outputs/encoded_data.jsonl'
model_dir = 'outputs/models'
output_dir = 'outputs/results'
test_size = 0.2
random_state = 42

# Create necessary directories
os.makedirs(os.path.dirname(encoded_data_file), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("Starting intent recognition pipeline...")

# 1. Data encoding
print("Step 1: Encoding data...")
encode_data_from_jsonl(
    input_file=data_file,
    output_file=encoded_data_file,
)

# 2. Model training
print("Step 2: Training model...")

# Load encoded data
embeddings, labels, _ = load_encoded_data(encoded_data_file)

# Prepare training and testing data
X_train, X_test, y_train, y_test, label_encoder = prepare_data(
    embeddings, labels, test_size=test_size, random_state=random_state
)

# Train model
model = train_model(
    X_train, y_train
)

# Save model
save_model(model, label_encoder, model_dir)

# 3. Model evaluation
print("Step 3: Evaluating model...")
results = evaluate_model(model, label_encoder, X_test, y_test)

# Save evaluation results
save_results(results, output_dir)

# Plot and save confusion matrix and classification report
plot_confusion_matrix(results, output_dir)
plot_classification_report(results, output_dir)

print("Intent recognition pipeline completed successfully!")
