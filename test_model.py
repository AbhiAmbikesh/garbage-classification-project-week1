import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model('garbage_classifier_model.h5')

# Garbage categories (same as training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load test image (replace this path with your image file)
img_path = 'test_image.jpg'  # <- yahan aap apna image file daalein
img = image.load_img(img_path, target_size=(128, 128))  # MUST match training size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"🧾 Predicted Garbage Type: {predicted_class}")
