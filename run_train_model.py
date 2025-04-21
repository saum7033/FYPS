from train_model import train_model
from run_load_data import train_features, train_labels
import numpy as np

# Total and extracted train data counts
total_train = 2201  # Total training data
extracted_train = len(train_features)  # Extracted training data (from features)

# Calculate train loss and accuracy
train_loss = (total_train - extracted_train) / total_train
train_accuracy = (extracted_train / total_train) * 100

# Display train loss and accuracy
print(f"Total Train Data: {total_train}")
print(f"Extracted Train Data: {extracted_train}")
print(f"Train Loss: {train_loss * 100:.2f}%")
print(f"Train Accuracy: {train_accuracy:.2f}%")

# Train the model and get metrics
accuracy, precision, loss = train_model(np.array(train_features), np.array(train_labels))

# Display the metrics
print(f"Training Accuracy: {accuracy * 100:.2f}%")
print(f"Training Precision: {precision * 100:.2f}%")
print(f"Training Loss: {loss:.4f}")