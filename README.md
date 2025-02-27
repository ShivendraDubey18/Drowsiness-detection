Driver Drowsiness Detection

Overview

This project is a Driver Drowsiness Detection System that uses a Convolutional Neural Network (CNN) to detect whether a driver is drowsy or alert based on real-time video feed. It utilizes OpenCV for image processing, TensorFlow/Keras for deep learning, and pygame for playing alerts.

Features

Real-time detection of drowsy and alert states.

Audio alert system to warn the driver when drowsiness is detected.

Pre-trained Haarcascade classifiers for face and eye detection.

Deep learning model (CNN) trained to recognize open and closed eyes.

Cross-platform support (Linux, Windows, MacOS).

Technologies Used

Python

OpenCV

TensorFlow/Keras

NumPy

Matplotlib

Pygame

Installation

Prerequisites

Ensure you have Python 3.7+ installed.

Step 1: Clone the Repository

git clone https://github.com/your-repo/Drowsiness-detection.git
cd Drowsiness-detection

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Download Haarcascade Files

Ensure the following files are in the project directory:

haarcascade_frontalface_alt.xml

haarcascade_lefteye_2splits.xml

haarcascade_righteye_2splits.xml

You can download them from the OpenCV GitHub repository.

Step 4: Train the Model (Optional)

To train the CNN model from scratch:

python model_training.py

This will generate models/cnncat2.h5.

Step 5: Run the Drowsiness Detection System

python drowsiness_detection.py

Usage

Ensure your webcam is connected.

Run the script, and the program will start detecting drowsiness.

If drowsiness is detected, an alarm will sound to alert the driver.

Press q to exit the program.

Model Architecture

The CNN model consists of:

Conv2D layers with ReLU activation.

MaxPooling layers to reduce feature dimensions.

BatchNormalization for stable training.

Dropout layers to prevent overfitting.

Flatten & Dense layers for classification.

License

This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments

OpenCV for pre-trained Haarcascade classifiers.

TensorFlow/Keras for deep learning capabilities.

Intel Open Source Computer Vision Library.

Contact

For any issues or contributions, reach out at: shivendrakumar3239@example.com or create an issue on GitHub.


