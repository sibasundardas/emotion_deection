import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st

# Use tf.keras.preprocessing.image for consistency
from tf_keras.preprocessing.image import img_to_array

# --- Load Model and Cascade Classifier ---
# Use a try-except block to gracefully handle loading errors
try:
    # Load the Haar Cascade for face detection
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Load the emotion detection model in the recommended .keras format
    # This is the key change to prevent segmentation faults.
    classifier = tf.keras.models.load_model('emotiondetector.keras')
except Exception as e:
    # Display a user-friendly error message if assets fail to load
    st.error(f"Error loading necessary assets: {e}")
    st.stop() # Stop the app if models can't be loaded

# Define the emotion labels corresponding to the model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# --- Video Transformer Class for Real-Time Processing ---
class EmotionDetector(VideoTransformerBase):
    """
    This class processes video frames from the webcam, detects faces,
    and predicts emotions for each detected face.
    """
    def transform(self, frame):
        # Convert the video frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convert the image to grayscale for the face detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Loop over each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Extract the region of interest (the face) from the grayscale image
            roi_gray = gray[y:y+h, x:x+w]
            
            # Resize the face to match the model's input size (48x48)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Preprocess the image for the model
            if np.sum([roi_gray]) != 0:
                # Normalize pixel values to be between 0 and 1
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                # Add a batch dimension to match the model's expected input shape
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction using the loaded model
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                
                # Position the text label above the rectangle
                label_position = (x, y - 10)
                
                # Draw the predicted emotion label on the image
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Return the processed image with rectangles and labels
        return img

# --- Streamlit User Interface ---
st.set_page_config(page_title="Real-Time Emotion Detector", page_icon="�")

st.title("Real-Time Emotion Recognition")
st.markdown("This app uses your webcam to detect faces and recognize emotions in real-time.")
st.markdown("Click **START** to begin and grant webcam access when prompted.")

# Start the WebRTC streamer to get video from the user's webcam
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown(
    """
    ---
    **How it works:**
    1. The app accesses your webcam stream.
    2. It detects faces in each frame using an OpenCV Haar Cascade.
    3. For each face, a deep learning model predicts the emotion.
    4. The detected emotion is displayed above the face.
    """
)