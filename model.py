"""
Driver Drowsiness Detection - Model Training
Copyright (C) 2025 [Shivendra Kumar Dubey (GTC)]
Licensed under the MIT License (https://opensource.org/licenses/MIT)
"""

import os
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization

# Define a generator function
def generator(directory, batch_size=32, target_size=(24, 24), class_mode='categorical'):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        directory, batch_size=batch_size, shuffle=True, color_mode='grayscale',
        class_mode=class_mode, target_size=target_size
    )

# Define batch size and target size
BATCH_SIZE = 32
TARGET_SIZE = (24, 24)
train_batch = generator('data/train', batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
valid_batch = generator('data/valid', batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

# Calculate steps per epoch
SPE = len(train_batch) // BATCH_SIZE
VS = len(valid_batch) // BATCH_SIZE
print(f"Training Steps per Epoch: {SPE}, Validation Steps: {VS}")

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Save the trained model
os.makedirs('models', exist_ok=True)
model.save('models/cnncat2.h5', overwrite=True)
print("Model saved successfully.")
