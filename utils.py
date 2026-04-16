"""
Physiosis — Utility functions and constants for gait analysis.
"""
import numpy as np
import mediapipe as mp

# ──────────────────────────────────────────────
# MediaPipe Pose landmark indices (0-32)
# ──────────────────────────────────────────────
LANDMARKS = mp.solutions.pose.PoseLandmark

# Left side
LEFT_SHOULDER = LANDMARKS.LEFT_SHOULDER
LEFT_HIP = LANDMARKS.LEFT_HIP
LEFT_KNEE = LANDMARKS.LEFT_KNEE
LEFT_ANKLE = LANDMARKS.LEFT_ANKLE
LEFT_HEEL = LANDMARKS.LEFT_HEEL
LEFT_FOOT_INDEX = LANDMARKS.LEFT_FOOT_INDEX

# Right side
RIGHT_SHOULDER = LANDMARKS.RIGHT_SHOULDER
RIGHT_HIP = LANDMARKS.RIGHT_HIP
RIGHT_KNEE = LANDMARKS.RIGHT_KNEE
RIGHT_ANKLE = LANDMARKS.RIGHT_ANKLE
RIGHT_HEEL = LANDMARKS.RIGHT_HEEL
RIGHT_FOOT_INDEX = LANDMARKS.RIGHT_FOOT_INDEX

# ──────────────────────────────────────────────
# Normal gait angle ranges (degrees)
# These are approximate clinical ranges for walking
# ──────────────────────────────────────────────
NORMAL_RANGES = {
    "hip": {"min": 100, "max": 180, "label": "Hip"},
    "knee": {"min": 90, "max": 180, "label": "Knee"},
    "ankle": {"min": 60, "max": 130, "label": "Ankle"},
}

# Symmetry threshold — difference above this is flagged
SYMMETRY_THRESHOLD = 15  # degrees

# Step detection — minimum Y-velocity change to register heel strike
HEEL_STRIKE_THRESHOLD = 0.015

# Minimum time between steps (seconds) — debounce for step detection
MIN_STEP_INTERVAL = 0.25

# ──────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle at point B formed by rays BA and BC.
    
    Args:
        a: First point [x, y] or [x, y, z]
        b: Vertex point [x, y] or [x, y, z]
        c: Third point [x, y] or [x, y, z]
    
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return round(angle, 1)


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_landmark_coords(landmarks, index, frame_w=1, frame_h=1):
    """
    Extract (x, y) pixel coordinates from a MediaPipe landmark.
    If frame_w/h are provided, coordinates are scaled to pixel space.
    """
    lm = landmarks[index]
    return np.array([lm.x * frame_w, lm.y * frame_h])


def classify_angle(angle: float, joint_name: str) -> str:
    """
    Classify an angle as 'normal', 'warning', or 'abnormal'
    based on clinical ranges.
    
    Returns: 'normal' | 'warning' | 'abnormal'
    """
    r = NORMAL_RANGES.get(joint_name)
    if r is None:
        return "normal"
    
    if r["min"] <= angle <= r["max"]:
        return "normal"
    
    # Within 10° of normal range → warning
    margin = 10
    if (r["min"] - margin) <= angle <= (r["max"] + margin):
        return "warning"
    
    return "abnormal"


def status_color(status: str):
    """Return BGR color tuple for a status level."""
    colors = {
        "normal": (0, 220, 0),       # Green
        "warning": (0, 220, 255),     # Yellow/Orange
        "abnormal": (0, 0, 255),      # Red
    }
    return colors.get(status, (200, 200, 200))
