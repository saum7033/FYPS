import cv2
import face_recognition
import numpy as np
import os

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

def process_and_display_image(image_path):
    print(f"Processing and displaying image: {image_path}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        window_name = 'Face Recognition'
        cv2.imshow(window_name, image)

        print("Please close the image window to proceed to the next image.")
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) == 27:  # ESC to break manually
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def recognize_faces():
    faces_folder = "Faces"
    print("Processing and displaying faces from the folder...")
    image_files = [f for f in os.listdir(faces_folder) if f.endswith(".jpg") or f.endswith(".png")]

    for filename in image_files:
        image_path = os.path.join(faces_folder, filename)
        process_and_display_image(image_path)

if __name__ == "__main__":
    try:
        # Proceed with recognizing and displaying faces
        recognize_faces()

    except KeyboardInterrupt:
        print("\nFace recognition interrupted by user.")
