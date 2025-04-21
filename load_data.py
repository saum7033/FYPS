import pandas as pd
import cv2
import os

def load_data(csv_file, image_folder):
    print(f"Reading CSV file: {csv_file}")
    data = pd.read_csv(csv_file)
    images1 = []
    images2 = []
    labels = []
    for index, row in data.iterrows():
        image1_path = os.path.join(image_folder, row['image1'])
        image2_path = os.path.join(image_folder, row['image2'])
        label = row['class']
        face_present = row['face_present']

        if face_present == 'Yes':
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            if image1 is None or image2 is None:
                print(f"Skipping row {index} due to missing images.")
                continue
            images1.append(image1)
            images2.append(image2)
            labels.append(label)
    print(f"Loaded {len(images1)} image pairs from {csv_file}")
    return images1, images2, labels