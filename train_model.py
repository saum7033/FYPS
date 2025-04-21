from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, hinge_loss
import joblib

def train_model(features, labels):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    clf = SVC(probability=True)  # Use an SVM classifier
    clf.fit(X_train, y_train)
    print("Model training completed.")

    # Save the trained model
    joblib.dump(clf, 'face_recognition_model.pkl')
    print("Trained model saved to face_recognition_model.pkl")

    # Evaluate on the validation set
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=1)

    # Calculate hinge loss
    y_val_scores = clf.decision_function(X_val)  # Get decision scores
    loss = hinge_loss(y_val, y_val_scores)

    return accuracy, precision, loss