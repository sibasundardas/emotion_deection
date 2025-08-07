import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading
import time
import queue

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# --- Caching for Model Loading ---
@st.cache_resource
def load_assets():
    """Loads and caches the face classifier and emotion model."""
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        emotion_model = tf.keras.models.load_model('emotiondetector.keras')
        return face_cascade, emotion_model
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

face_classifier, classifier = load_assets()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Emoji Mapping ---
emotion_emojis = {
    'Angry': '�',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😮'
}

# --- Video Transformer Class (with Queue for robust communication) ---
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        # A thread-safe queue to hold detection results
        self.result_queue = queue.Queue()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        latest_emotion = None

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                latest_emotion = label # Keep track of the last detected emotion
                
                label_position = (x, y - 10)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Put the latest detected emotion into the queue
        self.result_queue.put(latest_emotion)
        
        return img

# --- Streamlit UI Layout ---
if not face_classifier or not classifier:
    st.title("Real-Time Emotion Recognition")
    st.error("Failed to load necessary model files. Please ensure they are in the root directory.")
else:
    st.title("😊 Real-Time Emotion Recognition")
    st.markdown("This app uses your webcam to detect faces and recognize emotions in real-time.")
    st.markdown("Click **START** to begin and grant webcam access when prompted.")

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("---")
    st.header("Live Emotion Status")
    
    emotion_placeholder = st.empty()

    if webrtc_ctx.video_processor:
        while True:
            try:
                # Get the latest result from the queue
                result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                if result:
                    emoji = emotion_emojis.get(result, '')
                    emotion_placeholder.markdown(f"<h2 style='text-align: center;'>{result} {emoji}</h2>", unsafe_allow_html=True)
                # If result is None (no face detected), we can keep the last state or show detecting
                # For now, we do nothing to keep the last valid emotion on screen
            except queue.Empty:
                # If the queue is empty, it means no new frame has been processed
                # We can break or continue based on desired behavior
                if not webrtc_ctx.state.playing:
                    emotion_placeholder.markdown("<h2 style='text-align: center;'>Stopped</h2>", unsafe_allow_html=True)
                    break
    else:
        st.info("Click START to begin analysis.")