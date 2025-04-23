import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data
from feature_extraction import extract_features

def evaluate_model(test_features, test_labels):
    """
    Evaluate the pre-trained model using test features and labels.

    Args:
        test_features (array-like): Feature vectors for testing.
        test_labels (array-like): Corresponding labels.

    Returns:
        dict: Metrics including accuracy, precision, recall, and F1-score.
    """
    print("Loading the pre-trained classifier, scaler, and label encoder...")
    clf = joblib.load('face_recognition_model_xgb.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Classifier, scaler, and label encoder loaded successfully.")

    # Scale test features
    test_features = scaler.transform(test_features)

    # Encode test labels
    test_labels = label_encoder.transform(test_labels)

    # Predict and evaluate
    predictions = clf.predict(test_features)
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, average='weighted'),
        "recall": recall_score(test_labels, predictions, average='weighted'),
        "f1_score": f1_score(test_labels, predictions, average='weighted')
    }

    return metrics

if __name__ == "__main__":
    print("Loading test data...")
    csv_file = 'test.csv'
    image_folder = 'Faces'

    # Load test images and labels
    images1, images2, labels = load_data(csv_file, image_folder)

    # Extract features and filter labels
    features, labels = extract_features(images1, images2, labels)

    # Evaluate the model
    metrics = evaluate_model(features, labels)
    print(f"Metrics: {metrics}")