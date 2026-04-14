<div align="center">
  <h1>👁️ C-Vision: Unique People Counting & Recognition</h1>
  <p>
    <strong>A robust Computer Vision pipeline for tracking, counting, and analyzing unique individuals in real-time.</strong>
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
  [![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg?style=flat-square&logo=yolo)](https://github.com/ultralytics/ultralytics)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?style=flat-square&logo=opencv)](https://opencv.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
  
</div>

<hr />

## 📖 Overview

Welcome to the **Unique People Counting** project! This repository contains a state-of-the-art computer vision pipeline utilizing YOLOv8, Face Recognition, and Person Re-identification (ReID) architectures to accurately detect, track, count, and analyze individuals across video frames.

Whether you're analyzing retail footfall, securing an area, or studying movement patterns, this tool provides comprehensive tracking with features ranging from facial mapping to clothing analysis.

---

## ✨ Key Features

- **👤 Real-time Person & Face Detection**: Powered by YOLOv8 for lightning-fast and accurate bounding boxes.
- **🔢 Unique People Counting**: Advanced ReID and feature matching ensuring the same person isn't counted twice.
- **👕 Clothing Analysis**: Extract and classify apparel features to assist in secondary identification (`clothes_analyzer.py`).
- **🚀 High-performance Tracking**: Integration with ByteTrack and FAISS for fast similarity search and robust ID association.
- **💾 Database Management**: Simple storage for logging analyzed individuals over time (`analyze_db.py`).

---

## 🛠️ Tech Stack

| Technology | Usage |
| :--- | :--- |
| **YOLOv8** | Core Object & Face Detection models |
| **PyTorch & TorchReID** | Deep learning feature extraction |
| **ByteTrack** | Multi-object tracking |
| **FAISS** | Fast similarity search for Person ReID |
| **OpenCV** | Image processing and pipeline management |

---

## ⚙️ Installation

<details>
<summary><b>1. Clone the repository</b></summary>

```bash
git clone https://github.com/adityamishra139/Computer_vision_unique_people_count.git
cd Computer_vision_unique_people_count
```
</details>

<details>
<summary><b>2. Set up the Python Environment</b></summary>

It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
</details>


---

## 🚀 Usage

Make sure you have downloaded the required models (e.g., `yolov8n.pt`, `yolov8n-face.pt`, and caffemodels) and placed them in the root directory.

Run the main dashboard or script:

```bash
python main.py
```

### Environment Variables
For secure configurations (like API keys or specific DB paths), store them in a `.env` file at the root. *Note: `.env` is ignored by git.*

---

## 📂 Architecture & Structure

```text
📦 Computer_vision_unique_people_count
 ┣ 📂 Project_start/            # Initialization and starter scripts
 ┣ 📜 main.py                   # Main execution pipeline
 ┣ 📜 clothes_analyzer.py       # Clothing extraction module
 ┣ 📜 analyze_db.py             # Database handler
 ┣ 📜 FAISS_BYTETRACK_PATCH.patch # Tracker and search optimizations
 ┗ 📜 .gitignore                # Git ignored files configuration
```

---

<div align="center">
  <i>Developed with ❤️ for advancing computer vision analytics.</i>
</div>
