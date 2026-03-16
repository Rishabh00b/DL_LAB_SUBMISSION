import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ------------------------------
# Image size and batch size
# ------------------------------
img_size = 224
batch_size = 32

# ------------------------------
# Fix dataset path automatically
# ------------------------------
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "dataset/train")

print("Dataset Path:", data_dir)

# ------------------------------
# Data Preprocessing
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ------------------------------
# Training Generator
# ------------------------------
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# ------------------------------
# Validation Generator
# ------------------------------
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ------------------------------
# Build CNN Model
# ------------------------------
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))

# ------------------------------
# Compile Model
# ------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# Train Model
# ------------------------------
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# ------------------------------
# Evaluate Model
# ------------------------------
loss, accuracy = model.evaluate(val_generator)

print("Validation Accuracy:", accuracy)

# ------------------------------
# Plot Accuracy Graph
# ------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()