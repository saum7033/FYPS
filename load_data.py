import pandas as pd
import cv2
import os

def load_data(csv_file, image_folder):
    """
    Load image pairs and their labels from a CSV file and an image folder.

    Args:
        csv_file (str): Path to the CSV file containing image metadata.
        image_folder (str): Path to the folder containing the images.

    Returns:
        tuple: Three lists - images1, images2, and labels.
    """
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

if __name__ == "__main__":
    # Use train.csv and test.csv in the root directory
    train_csv = 'train.csv'  # Path to train.csv in the root directory
    test_csv = 'test.csv'    # Path to test.csv in the root directory
    image_folder = 'Faces'   # Path to the folder containing images

    # Load training data
    print("Loading training data...")
    train_images1, train_images2, train_labels = load_data(train_csv, image_folder)
    print(f"Training data: {len(train_images1)} image pairs and {len(train_labels)} labels.")

    # Load testing data
    print("Loading testing data...")
    test_images1, test_images2, test_labels = load_data(test_csv, image_folder)
    print(f"Testing data: {len(test_images1)} image pairs and {len(test_labels)} labels.")