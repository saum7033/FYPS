import numpy as np
import face_recognition

def extract_features(images1, images2):
    features = []
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        print(f"Extracting features for image pair {i}")
        face_encodings1 = face_recognition.face_encodings(image1)
        face_encodings2 = face_recognition.face_encodings(image2)
        if face_encodings1 and face_encodings2:
            combined_features = np.concatenate((face_encodings1[0], face_encodings2[0]))
            features.append(combined_features)
        else:
            print(f"Skipping image pair {i} due to missing face encodings.")
    print(f"Extracted features for {len(features)} image pairs")
    return features