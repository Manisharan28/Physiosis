# Physiosis — Real-Time Gait Analysis System

Physiosis is a real-time physiotherapy tool that analyzes human gait using computer vision. It leverages MediaPipe Pose and OpenCV to track body movements, calculate joint angles, and detect abnormalities during walking.

---

## 🚀 Features

* 🎥 **Dual Input**: Works with webcam or recorded videos
* 🦵 **Joint Tracking**: Hip, knee, and ankle angles (both sides)
* 👣 **Gait Cycle Detection**: Step counting and cadence (steps/min)
* ⚖️ **Symmetry Analysis**: Left vs right comparison
* ⚠️ **Abnormality Detection**: Flags deviations using thresholds
* 📊 **Gait Quality Score**: Overall score (0–100)
* 📺 **Live Overlay Dashboard**: Real-time metrics on video

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe

---

## 📁 Project Structure

```
physiosis/
 ├── src/
 │   ├── main.py
 │   ├── gait_analyzer.py
 │   ├── video_handler.py
 │   ├── overlay.py
 │   └── utils.py
 │
 ├── templates/
 │   └── index.html
 │
 ├── static/
 │   ├── css/
 │   └── js/
 │
 ├── requirements.txt
 ├── README.md
 ├── .gitignore
 └── LICENSE
```

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

---

## ▶️ Usage

```
# Run with webcam
python -m src.main

# Run with video file
python -m src.main --source path/to/video.mp4

# Custom settings
python -m src.main --source 0 --width 1280 --height 720 --confidence 0.7
```

---

## 🎮 Controls

| Key | Action          |
| --- | --------------- |
| Q   | Quit            |
| R   | Reset metrics   |
| S   | Save screenshot |
| P   | Pause / Resume  |

---

## 📊 Analysis Parameters

| Parameter    | Description                |
| ------------ | -------------------------- |
| Hip Angle    | Shoulder → Hip → Knee      |
| Knee Angle   | Hip → Knee → Ankle         |
| Ankle Angle  | Knee → Ankle → Foot        |
| Symmetry     | Left vs right difference   |
| Cadence      | Steps per minute           |
| Gait Quality | Combined performance score |

---

## 💡 Future Improvements

* Add ML-based posture correction suggestions
* Store session history and analytics
* Web dashboard for remote monitoring

---

## 📄 License

This project is licensed under the Apache 2.0 License.
