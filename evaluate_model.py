import joblib
from sklearn.metrics import accuracy_score, precision_score

def evaluate_model(test_features, test_labels, total_test):
    """Evaluate the model using the provided test features and labels."""
    # Load the pre-trained classifier
    print("Loading the pre-trained classifier...")
    clf = joblib.load('face_recognition_model.pkl')
    print("Classifier loaded successfully.")

    # Print shapes of test features and labels
    print(f"Shape of test features: {len(test_features)}")
    print(f"Shape of test labels: {len(test_labels)}")
    print(f"Total Test Data: {total_test}")
    print(f"Extracted Test Data: {len(test_features)}")

    # Handle mismatch between test features and labels
    if len(test_features) != len(test_labels):
        print("Warning: Mismatch between test features and labels.")
        min_length = min(len(test_features), len(test_labels))
        test_features = test_features[:min_length]
        test_labels = test_labels[:min_length]
        print(f"Using first {min_length} samples for evaluation.")

    # Predict the labels for the test features
    predictions = clf.predict(test_features)

    # Calculate accuracy and precision
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=1)

    # Calculate test loss
    test_loss = (total_test - len(test_features)) / total_test

    # Display the results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Test Loss: {test_loss * 100:.2f}%")

if __name__ == "__main__":
    try:
        # Example usage of the evaluate_model function
        # Replace with actual test features and labels
        test_features = []  # Load or define test features
        test_labels = []    # Load or define test labels
        total_test = 100    # Define total test data count
        evaluate_model(test_features, test_labels, total_test)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")