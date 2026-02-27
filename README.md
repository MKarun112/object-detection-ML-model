🚀 Real-Time Object Detection using SSD-MobileNet V3 & OpenCV DNN
📌 Overview

This project implements a real-time multi-object detection system using OpenCV’s Deep Neural Network (DNN) module with a pre-trained SSD-MobileNet V3 (COCO) model.

The system performs object detection on:

Static images

Pre-recorded video files

Live webcam streams

It is optimized for high-speed inference while maintaining reliable detection accuracy across 80+ object classes.

🧠 Model Architecture
🔹 Backbone: MobileNet V3

Lightweight convolutional neural network designed for efficient feature extraction with low computational overhead.

🔹 Detector: SSD (Single Shot MultiBox Detector)

Performs object localization and classification in a single forward pass, enabling real-time detection.

🔹 Dataset: COCO (Common Objects in Context)

Pre-trained on 80 object categories including:

person, car, bicycle, laptop, dog, chair, bottle, bus, truck, traffic light and more.

Model files used:

frozen_inference_graph.pb → Pre-trained weights

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt → Network configuration

labels.txt → COCO class labels

⚙️ System Architecture

Input (Image / Video / Webcam)
→ Frame Preprocessing
→ DNN Forward Pass
→ Bounding Box Extraction
→ Confidence Filtering
→ Label Mapping
→ Visualization Output

🔬 Technical Implementation
1️⃣ Model Initialization
model = cv2.dnn_DetectionModel(frozen_model, config_file)

Configured with:

Input Size: 320 × 320

Input Scale: 1 / 127.5 (Normalization)

Mean Subtraction: (127.5, 127.5, 127.5)

Channel Swap: BGR → RGB

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

These preprocessing steps ensure compatibility with the TensorFlow-trained SSD-MobileNet model.

2️⃣ Object Detection

Detection is performed using:

ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

ClassIndex → Detected object class IDs

confidence → Detection confidence scores

bbox → Bounding box coordinates

Confidence threshold tuning (0.5–0.55) reduces false positives.

3️⃣ Visualization Pipeline

For each detected object:

Draw bounding box using cv2.rectangle()

Overlay class label using cv2.putText()

Render output in real-time using cv2.imshow()

Bounding box format:

(x, y, width, height)
📹 Real-Time Video Processing

Implemented frame-by-frame detection loop.

Supports fallback from video file to webcam (cv2.VideoCapture(0)).

Handles dynamic object tracking across frames.

Exit condition implemented using keyboard interrupt ('q').

🚀 Key Features

✔ Real-time multi-object detection
✔ 80+ COCO object classes
✔ Lightweight inference (MobileNet V3 backbone)
✔ Confidence threshold filtering
✔ Image, video, and webcam compatibility
✔ Optimized preprocessing pipeline
✔ Efficient bounding box rendering

🛠 Tech Stack

Python 3.x

OpenCV (cv2 – DNN module)

SSD-MobileNet V3

COCO Dataset

Matplotlib (for image visualization)

📊 Performance Characteristics

Fast inference due to lightweight backbone

Suitable for CPU-based real-time detection

Balanced speed vs. accuracy trade-off

Minimal external deep learning dependencies

📚 Learning Outcomes

Deployment of pre-trained deep learning models

OpenCV DNN module usage

Image preprocessing techniques

Real-time video frame processing

Bounding box coordinate handling

Confidence score filtering

Computer Vision system pipeline design

🔮 Future Improvements

Integrate object tracking (e.g., SORT / DeepSORT)

Add FPS counter overlay

Enable GPU acceleration (CUDA support)

Implement custom-trained object detection model

Convert to Flask/Streamlit web application

Add confidence score display per object

📎 Example Use Cases

Surveillance systems

Smart traffic monitoring

Retail analytics

Assistive vision systems

Automated inventory detection

🏁 Conclusion

This project demonstrates practical implementation of a real-time deep learning-based object detection system using OpenCV without relying on heavy training frameworks during inference. It highlights strong understanding of computer vision pipelines, model deployment, and real-time processing systems.
