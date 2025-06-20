import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Dataset folder path (update with your path)
train_path = "dataset/train"
val_path = "dataset/validation"

# ImageDataGenerator for loading and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_path, target_size=(128,128), batch_size=32, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_path, target_size=(128,128), batch_size=32, class_mode='categorical')

# Model Build
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, epochs=5, validation_data=val_data)

# Save
model.save("garbage_classifier_model.h5")

print("✅ Model trained and saved.")
