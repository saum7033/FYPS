import pandas as pd
import joblib
import cv2
import os
from feature_extraction import extract_features

def create_face_database(csv_file, output_pkl):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Initialize the database dictionary
    database = {}

    for index, row in data.iterrows():
        image_path = row['image1']
        name = image_path.split('_')[0]  # Use the first part of the filename as the name
        full_image_path = os.path.join('Faces', image_path)  # Update this path as needed
        image = cv2.imread(full_image_path)
        if image is not None:
            features = extract_features([image], [image])
            if features:
                database[name] = features[0][0]
            else:
                print(f"Could not extract features for {name} from {full_image_path}")
        else:
            print(f"Could not read image {full_image_path}")

    # Save the database to a pickle file
    joblib.dump(database, output_pkl)
    print(f"Face database saved to {output_pkl}")

if __name__ == "__main__":
    create_face_database('test.csv', 'face_database.pkl')  # Updated file path