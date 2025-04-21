# Import the recognize_faces function from the real_time_recognition_script module
from real_time_recognition_script import recognize_faces

# Call the recognize_faces function to start the real-time face recognition process
# This function will:
# 1. Load the pre-trained face recognition model.
# 2. Load the database of pre-stored face features.
# 3. Initialize the webcam to capture real-time video.
# 4. Continuously capture frames from the webcam.
# 5. Extract features from each frame using the extract_features_from_image function.
# 6. Use the classifier to predict if the face in the frame is similar to the stored faces.
# 7. Compare the extracted features with the database to find the closest match.
# 8. Display the result (similar/not similar and matched label) on the video frame.
# 9. Break the loop and close the webcam if the 'q' key is pressed.
# python3 create_face_database.py
# python3 run_train_model.py
# python3 run_evaluate_model.py
# python3 run_real_time_recognition.py



recognize_faces()