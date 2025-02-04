Face Recognition System Using MobileNetV2 on ESP32-S3

This repository contains a face recognition system built using MobileNetV2, optimized for deployment on ESP32-S3 via PlatformIO. The project involves training the model, generating an INT8 quantized version, and integrating it with ESP32-S3 for efficient edge inference.

🚀 Features

Preprocessing: Converts images to RGB, resizes to 128x128 pixels, and normalizes pixel values.

Data Augmentation: Includes random rotations, shifts, shears, zooms, and flips to improve model generalization.

Model Architecture: Uses MobileNetV2 as the base model with a dropout layer and softmax output for classification.

Evaluation: Provides test accuracy, classification reports, and latency measurements.

Quantization: Converts the trained model to TFLite format with INT8 quantization for efficient inference.

ESP32-S3 Integration: Deploys the quantized model to ESP32-S3 for edge-based face recognition.

🛠️ Prerequisites

Ensure the following are installed:

Python 3.7+

TensorFlow 2.x

NumPy

scikit-learn

PIL (Pillow)

PlatformIO

ESP-IDF (for ESP32-S3)

📂 Dataset

Place your face image dataset in the dataset directory. Ensure images are named in the format image_<class>.jpg, where <class> is the class index starting from 1.

Build your own dataset.

Aim for more than 1000 images in different views and lighting conditions.

⚡ How to Run

Step 1: Clone the Repository

git clone https://github.com/<username>/face-recognition-esp32s3.git
cd face-recognition-esp32s3

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Train the Model

The training scripts are available in the Face_Recognition folder.

Run the code in Google Colab for faster training.

Install PlatformIO in VS Code to deploy the trained model to ESP32-S3.

Step 4: Deploy to ESP32-S3

Set up PlatformIO and create a new project for ESP32-S3.

Copy the model-int8.cc file to the data directory of your PlatformIO project.

Use the TensorFlow Lite Micro library to load and run the model on ESP32-S3.

Flash the project to your ESP32-S3 device:

platformio run --target upload

📊 Performance Metrics

Test Accuracy: Displayed during model evaluation.

Model Size: Approximately 2.5 MB for the original model; smaller for the quantized model.

Latency: Measured and printed for a single inference.

Classification Report: Includes precision, recall, and F1-score.

🤝 Contact

For implementation assistance or further questions, feel free to reach out!

