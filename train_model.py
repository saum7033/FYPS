from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from xgboost import XGBClassifier
from load_data import load_data
from feature_extraction import extract_features

def train_model(features, labels):
    """
    Train an XGBoost model and evaluate it on a validation set.

    Args:
        features (array-like): Feature vectors for training.
        labels (array-like): Corresponding labels.

    Returns:
        dict: Metrics including accuracy, precision, recall, and F1-score.
    """
    # Encode string labels into numeric values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    print("Model training completed.")

    # Save the trained model, scaler, and label encoder
    joblib.dump(clf, 'face_recognition_model_xgb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Trained model, scaler, and label encoder saved.")

    # Evaluate the model
    y_val_pred = clf.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, average='weighted'),
        "recall": recall_score(y_val, y_val_pred, average='weighted'),
        "f1_score": f1_score(y_val, y_val_pred, average='weighted')
    }

    return metrics

if __name__ == "__main__":
    print("Loading data...")
    csv_file = 'train.csv'
    image_folder = 'Faces'

    # Load images and labels
    images1, images2, labels = load_data(csv_file, image_folder)

    # Extract features and filter labels
    features, labels = extract_features(images1, images2, labels)

    # Train the model
    metrics = train_model(features, labels)
    print(f"Metrics: {metrics}")