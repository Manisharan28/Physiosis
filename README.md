<<<<<<< HEAD
# Physiosis — Real-Time Physiotherapy Gait Analysis

A Python-based gait analysis tool for physiotherapy that uses **MediaPipe Pose** and **OpenCV** to detect human movement from a webcam or recorded video, analyze gait parameters in real time, and flag abnormalities on-screen.

## Features

- **Dual Input**: Live webcam feed or pre-recorded video files (MP4, AVI, etc.)
- **Real-Time Joint Tracking**: Hip, knee, and ankle angles (both sides)
- **Gait Cycle Detection**: Step counting, cadence (steps/min)
- **Symmetry Analysis**: Left/right comparison with visual bars
- **Abnormality Detection**: Color-coded warnings for deviations from normal ranges
- **Quality Scoring**: Overall gait quality score (0-100)
- **Live Dashboard**: Side panel with all metrics overlaid on video

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (default)
python main.py

# Recorded video
python main.py --source path/to/walking_video.mp4

# Custom display size and confidence
python main.py --source 0 --width 1280 --height 720 --confidence 0.7
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset all metrics |
| `S` | Save screenshot |
| `P` | Pause / Resume |

## Architecture

```
main.py           → Entry point, CLI, main loop
gait_analyzer.py  → Core engine: angles, cycles, scoring
video_handler.py  → Webcam & video file input
overlay.py        → Skeleton drawing, metrics panel, warnings
utils.py          → Math helpers, constants, thresholds
```

## Analysis Parameters

| Parameter | What it measures |
|-----------|-----------------|
| Hip Angle | Shoulder → Hip → Knee flexion/extension |
| Knee Angle | Hip → Knee → Ankle flexion/extension |
| Ankle Angle | Knee → Ankle → Foot dorsiflexion |
| Symmetry | Left vs. Right angle difference per joint |
| Cadence | Steps per minute |
| Gait Quality | Composite score from angle normality + symmetry |
=======
# Physiosis
>>>>>>> a2530468e0d556999fdf88deb1c3419513f6195c
