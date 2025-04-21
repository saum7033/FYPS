import os
import cv2
import face_recognition

INPUT_DIR = "./Faces"
OUTPUT_SINGLE = "./Faces_Single"
OUTPUT_MULTI = "./Faces_Multi"

# Create output directories if they don't exist
os.makedirs(OUTPUT_SINGLE, exist_ok=True)
os.makedirs(OUTPUT_MULTI, exist_ok=True)

def detect_and_categorize_faces():
    for filename in os.listdir(INPUT_DIR):
        # Skip non-image files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load {filename}")
            continue

        # Convert the image to RGB for face detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)

        # Draw green rectangles around detected faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Copy the processed image to the appropriate folder
        if len(face_locations) >= 2:
            dest_path = os.path.join(OUTPUT_MULTI, filename)
        else:
            dest_path = os.path.join(OUTPUT_SINGLE, filename)

        cv2.imwrite(dest_path, image)  # Save the processed image
        print(f"{filename}: {len(face_locations)} face(s) detected and copied to {dest_path}")

if __name__ == "__main__":
    detect_and_categorize_faces()