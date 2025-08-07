import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# This is the corrected import line
from tensorflow.keras.preprocessing.image import img_to_array

# --- Function to Load Model and Classifier (with Caching) ---
@st.cache_resource
def load_emotion_model():
    """
    Loads the face classifier and the emotion model into memory and caches them.
    """
    face_clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_clf = tf.keras.models.load_model('emotiondetector.keras')
    return face_clf, emotion_clf

# Load the assets using the cached function
try:
    face_classifier, classifier = load_emotion_model()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Define the emotion labels corresponding to the model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# --- Video Transformer Class for Real-Time Processing ---
class EmotionDetector(VideoTransformerBase):
    """
    This class processes video frames from the webcam, detects faces,
    and predicts emotions for each detected face.
    """
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# --- Streamlit User Interface ---
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
    2. It detects faces in each frame using an OpenCV Haar Cascade.
    3. For each face, a deep learning model predicts the emotion.
    4. The detected emotion is displayed above the face.
    """
)
