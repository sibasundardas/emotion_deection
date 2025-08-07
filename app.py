import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st

# Load model and cascade classifier
try:
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('emotiondetector.h5')
except Exception as e:
    st.error(f"Error loading assets: {e}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Preprocess the face for the model
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                
                # Add the label to the image
                label_position = (x, y - 10)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# Streamlit UI
st.set_page_config(page_title="Real-Time Emotion Detector", page_icon="😊")

st.title("Real-Time Emotion Recognition")
st.markdown("This app uses your webcam to detect faces and recognize emotions in real-time.")
st.markdown("Click **START** to begin and grant webcam access when prompted.")

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
    2. It detects faces in each frame using an OpenCV Cascade Classifier.
    3. For each face, a deep learning model (trained with Keras/TensorFlow) predicts the emotion.
    4. The detected emotion is displayed above the face.
    """
)