AI Crowd Monitor & Smart Surveillance System 👁️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Large-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An autonomous, real-time crowd detection and density monitoring system built using state-of-the-art Computer Vision (YOLOv8) and OpenCV. Designed for smart surveillance, event management, and public safety.


🚀 Key Features

* **High-Accuracy Detection:** Utilizes the heavy-weight **YOLOv8 Large (`yolov8l.pt`)** model for detecting humans at long ranges and in dense environments.
* **Real-Time Density Monitoring:** Actively tracks the total human count in the video stream.
* **Automated Safety Alerts:** Implements logic to trigger a high-visibility **"OVERCROWDING DETECTED"** alert when a predefined threshold is crossed.
* **High-Resolution Input Processing:** Scales input streams to 1280px to preserve small-object details in 4K video feeds.
* **Selective Tracking:** The model is aware of 80 distinct object classes but is programmed to isolate and track **humans only** for density calculations.


 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV (cv2), cvzone
* **Deep Learning Model:** Ultralytics YOLOv8 (Large Weights)
* **IDE:** AntiGravity / VS Code


🏗️ System Architecture 

1. **Input Feed:** Captures live webcam feed or pre-recorded high-resolution video (e.g., 4K 30fps).
2. **AI Processing:** Passes frames through the YOLOv8l Neural Network.
3. **Filtering Logic:** Filters out non-human detections (bicycles, cars, etc.).
4. **Data Overlay:** Draws dynamic bounding boxes (Green = Safe, Red = Alert) and overlays real-time metrics.

## 💻 How to Run

**1. Clone the repository**

git clone [https://github.com/YOUR_USERNAME/AI-Crowd-Monitor.git](https://github.com/YOUR_USERNAME/AI-Crowd-Monitor.git)
cd AI-Crowd-Monitor

**2. Install dependencies**
   
pip install ultralytics opencv-python cvzone
    
**3. Run the script**

python crowd_monitor.py

💡 Real-World Applications

This system is designed as a foundational module for:

Smart Cities & Transport: Monitoring crowd levels at train stations or bus terminals.

Event Safety: Preventing stampedes at religious gatherings, festivals, and concerts.

Retail Analytics: Tracking footfall and queue lengths in shopping centers.

🔮 Future Upgrades Roadmap

[ ] Region of Interest (ROI): Define restricted zones for targeted counting.

[ ] Data Logging: Export timestamped crowd density data to an Excel/CSV file.

[ ] Directional Tracking: Detect if people are entering or exiting an area.


