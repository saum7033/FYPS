import numpy as np
import face_recognition

def extract_features(images1, images2, labels):
    """
    Extract face embeddings for pairs of images and combine them into a single feature vector.

    Args:
        images1 (list): List of first images in the pairs.
        images2 (list): List of second images in the pairs.
        labels (list): List of labels corresponding to the image pairs.

    Returns:
        tuple: Combined feature vectors and filtered labels.
    """
    features = []
    filtered_labels = []
    for i, (image1, image2, label) in enumerate(zip(images1, images2, labels)):
        print(f"Extracting features for image pair {i + 1}")
        face_encodings1 = face_recognition.face_encodings(image1)
        face_encodings2 = face_recognition.face_encodings(image2)
        if face_encodings1 and face_encodings2:
            combined_features = np.concatenate((face_encodings1[0], face_encodings2[0]))
            features.append(combined_features)
            filtered_labels.append(label)  # Keep the label only if features are extracted
        else:
            print(f"Skipping image pair {i + 1} due to missing face encodings.")
    print(f"Extracted features for {len(features)} image pairs.")
    return features, filtered_labels