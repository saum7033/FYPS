from load_data import load_data
from feature_extraction import extract_features
import numpy as np

print("Loading training data...")
train_images1, train_images2, train_labels = load_data('train.csv', 'Faces')  # Updated file path
print("Training data loaded.")

print("Loading testing data...")
test_images1, test_images2, test_labels = load_data('test.csv', 'Faces')  # Updated file path
print("Testing data loaded.")

print("Extracting features from training data...")
train_features = extract_features(train_images1, train_images2)
print("Features extracted from training data.")

print("Extracting features from testing data...")
test_features = extract_features(test_images1, test_images2)
print("Features extracted from testing data.")

print(f"Shape of training features: {np.array(train_features).shape}")
print(f"Shape of testing features: {np.array(test_features).shape}")