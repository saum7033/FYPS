import pandas as pd
import joblib
import cv2
import os
from tqdm import tqdm  # For progress tracking
from feature_extraction import extract_features

def create_face_database(csv_file, output_pkl):
    """
    Create a face database by extracting features from images.

    Args:
        csv_file (str): Path to the CSV file containing image metadata.
        output_pkl (str): Path to save the face database as a pickle file.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Initialize the database dictionary
    database = {}

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing images"):
        image_path = row['image1']
        name = image_path.split('_')[0]  # Use the first part of the filename as the name
        full_image_path = os.path.join('Faces', image_path)  # Update this path as needed
        image = cv2.imread(full_image_path)
        if image is not None:
            # Pass a dummy label since labels are not needed for the database
            features, _ = extract_features([image], [image], [0])  # Dummy label
            if features:
                database[name] = features[0]
            else:
                print(f"Could not extract features for {name}")
        else:
            print(f"Could not read image {full_image_path}")

    # Validate the database
    if len(database) == 0:
        print("No features were extracted. Please check the input data.")
        return

    # Save the database to a pickle file
    joblib.dump(database, output_pkl)
    print(f"Face database saved to {output_pkl} with {len(database)} entries.")

if __name__ == "__main__":
    create_face_database('test.csv', 'face_database.pkl')