import tensorflow as tf

# Load your old model
model = tf.keras.models.load_model('emotiondetector.h5')

# Save it in the new, recommended format
model.save('emotiondetector.keras')

print("Successfully converted model to emotiondetector.keras")