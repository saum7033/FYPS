import cv2
import face_recognition
import numpy as np
import os

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

def recognize_faces():
    faces_folder = "Faces"
    image_files = [f for f in os.listdir(faces_folder) if f.endswith(('.jpg', '.png'))]

    for filename in image_files:
        image_path = os.path.join(faces_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)

        for (top, right, bottom, left) in face_locations:
            # Draw a green rectangle around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Face Recognition', image)

        # Wait for the user to close the window or press a key
        while True:
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break  # Window closed by the user
            if key == 27:  # ESC key pressed
                break

        # Close the current window before moving to the next image
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()