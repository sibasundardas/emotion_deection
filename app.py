import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading
import time

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
# A dictionary to map emotion labels to emojis
emotion_emojis = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😮'
}

# --- Shared State for Emotion Prediction ---
# This thread-safe container will hold the latest detected emotion
lock = threading.Lock()
latest_prediction_container = {"emotion": None}

# --- Video Transformer Class ---
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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
                
                # Update the shared container with the latest emotion
                with lock:
                    latest_prediction_container["emotion"] = label
                
                # Draw the text label on the video frame
                label_position = (x, y - 10)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
        video_transformer_factory=EmotionDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("---")
    st.header("Live Emotion Status")
    
    # Create a placeholder for the emoji and text
    emotion_placeholder = st.empty()

    # Continuously update the placeholder while the video is playing
    while webrtc_ctx.state.playing:
        with lock:
            emotion = latest_prediction_container["emotion"]
        
        if emotion:
            emoji = emotion_emojis.get(emotion, '')
            emotion_placeholder.markdown(f"<h2 style='text-align: center;'>{emotion} {emoji}</h2>", unsafe_allow_html=True)
        else:
            emotion_placeholder.markdown("<h2 style='text-align: center;'>Detecting...</h2>", unsafe_allow_html=True)
            
        time.sleep(0.1) # Update every 100ms
        
