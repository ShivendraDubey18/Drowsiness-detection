"""
Driver Drowsiness Detection System
Copyright (C) 2025 [Shivendra Kumar Dubey (GTC)]
Licensed under the MIT License (https://opensource.org/licenses/MIT)

This software uses pre-trained Haarcascade classifiers from OpenCV,
which are distributed under the Intel Open Source Computer Vision Library License.
For more details, see:
https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

import cv2
import os
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
    exit()
from pygame import mixer
import time

# Initialize pygame mixer for alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Validate Haarcascade classifier paths
def load_cascade(path):
    if os.path.exists(path):
        return cv2.CascadeClassifier(path)
    else:
        print(f"Error: Cascade file {path} not found. Ensure it exists.")
        exit()

# Load OpenCV Haarcascade files (licensed under Intel Open Source Computer Vision Library License)
face = load_cascade('haarcascade_frontalface_alt.xml')
leye = load_cascade('haarcascade_lefteye_2splits.xml')
reye = load_cascade('haarcascade_righteye_2splits.xml')

# Load the trained model
model_path = 'models/cnncat2.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print(f"Error: Model file {model_path} not found. Ensure it exists.")
    exit()

# Label categories
labels = ['Closed', 'Open']

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count, score, thicc = 0, 0, 2
rpred, lpred = [99], [99]

def predict_eye(eye):
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    eye = eye / 255.0
    eye = np.reshape(eye, (1, 24, 24, 1))
    return np.argmax(model.predict(eye))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        rpred[0] = predict_eye(frame[y:y + h, x:x + w])
        break

    for (x, y, w, h) in left_eye:
        lpred[0] = predict_eye(frame[y:y + h, x:x + w])
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    score = max(score, 0)
    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        sound.play()
        thicc = min(thicc + 2, 16)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    
    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
