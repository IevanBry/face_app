# Face Emotion Detection App 😄😡😢

A Flutter application for real-time **facial emotion detection** using the **YOLO (You Only Look Once)** object detection model.

## 📱 About the App

This app uses the device’s camera to detect and classify human facial emotions in real time. It supports recognizing common facial expressions such as:

- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😮 Surprised  
- 😐 Neutral

### 🧠 Key Technologies
- **Flutter** – Cross-platform mobile development framework.
- **YOLO (TFLite/ONNX)** – Real-time object detection and emotion classification.
- **camera** package – For accessing the device's camera.
- **image** package – Image preprocessing.
- **tflite / yolov5_flutter / tflite_flutter** – For running machine learning models on-device.

## 📦 Features
- face detection from camera input.
- Emotion classification using YOLO-based model.
- Overlay bounding boxes and emotion labels on detected faces.
- Supports multiple faces in a frame.
