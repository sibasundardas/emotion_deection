import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import queue
from PIL import Image, ImageDraw, ImageFont

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# --- Caching for Model Loading ---
@st.cache_resource
def load_assets():
    """Loads and caches the face classifier, emotion model, and emoji font."""
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        emotion_model = tf.keras.models.load_model('emotiondetector.keras')
        # Load the emoji font. Size 32 is a good starting point.
        emoji_font = ImageFont.truetype("emoji_font.ttf", 32)
        return face_cascade, emotion_model, emoji_font
    except Exception as e:
        st.error(f"Error loading assets: {e}. Make sure emoji_font.ttf is in your project folder.")
        return None, None, None

face_classifier, classifier, font = load_assets()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Emoji Mapping ---
emotion_emojis = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
    'Neutral': '😐', 'Sad': '😢', 'Surprise': '😮'
}

# --- Video Transformer Class ---
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert OpenCV BGR image to Pillow RGB image to draw text
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle using Pillow
            draw.rectangle(((x, y), (x + w, y + h)), outline="lime", width=3)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emoji = emotion_emojis.get(label, "")
                
                # Create the text with emoji
                display_text = f"{label} {emoji}"
                
                # Draw the text and emoji on the image using Pillow
                label_position = (x, y - 40) # Adjust position for bigger font
                draw.text(label_position, display_text, font=font, fill=(0, 255, 0))
        
        # Convert back to OpenCV BGR format
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Streamlit UI Layout ---
if not all([face_classifier, classifier, font]):
    st.title("Real-Time Emotion Recognition")
    st.error("Application failed to load necessary assets. Please check the logs.")
else:
    st.title("😊 Real-Time Emotion Recognition")
    st.markdown("Click **START** to begin emotion detection.")

    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
