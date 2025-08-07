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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a 'Cool' UI ---
st.markdown("""
    <style>
        /* General App Styling */
        .stApp {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        
        /* Main Title */
        h1 {
            color: #e94560;
            text-align: center;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }

        /* Subheaders and Markdown */
        h2, h3, .stMarkdown {
            color: #f0f0f0;
        }

        /* Video Container Styling */
        div[data-testid="stVideo"] {
            border-radius: 15px;
            overflow: hidden;
            border: 2px solid #e94560;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }

        /* Results Panel Styling */
        div[data-testid="stVerticalBlock"] .st-emotion-cache-1r6slb0 {
             background-color: #16213e;
             border-radius: 15px;
             padding: 20px;
             border: 1px solid #0f3460;
        }
        
        /* Progress Bar for Confidence */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #16213e, #e94560);
        }

    </style>
""", unsafe_allow_html=True)


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

# Thread-safe dictionary to store the latest prediction
lock = threading.Lock()
latest_prediction_container = {"emotion": None, "confidence": 0.0}

# --- Video Transformer Class ---
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (233, 69, 96), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                confidence = np.max(prediction)
                label = emotion_labels[prediction.argmax()]

                with lock:
                    latest_prediction_container["emotion"] = label
                    latest_prediction_container["confidence"] = confidence
                
                label_position = (x, y - 10)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (233, 69, 96), 2)
        
        return img

# --- Streamlit UI Layout ---
if not face_classifier or not classifier:
    st.title("Real-Time Emotion Recognition")
    st.error("Failed to load necessary model files. Please ensure they are in the root directory.")
else:
    st.title("😊 Real-Time Emotion Recognition")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=EmotionDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.header("Analysis Results")
        results_placeholder = st.empty()

    while webrtc_ctx.state.playing:
        with lock:
            emotion = latest_prediction_container["emotion"]
            confidence = latest_prediction_container["confidence"]
        
        with results_placeholder.container():
            if emotion:
                st.subheader(f"Detected Emotion: {emotion}")
                st.write("Confidence:")
                st.progress(confidence)
                
                # Display all probabilities
                st.write("---")
                st.write("Emotion Probabilities:")
                # You can create a small chart or just list them
                for i, label in enumerate(emotion_labels):
                     st.write(f"{label}: {classifier.predict(np.zeros((1,48,48,1)))[0][i]:.2%}")


            else:
                st.info("No face detected or analysis hasn't started yet.")
        
        time.sleep(0.1)
