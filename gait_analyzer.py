"""
Physiosis — Core gait analysis engine.
Computes joint angles, detects gait cycles, and scores movement quality.
"""
import time
import numpy as np
from collections import deque

from utils import (
    calculate_angle,
    get_landmark_coords,
    classify_angle,
    NORMAL_RANGES,
    SYMMETRY_THRESHOLD,
    HEEL_STRIKE_THRESHOLD,
    LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX,
    RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX,
)


class GaitAnalyzer:
    """
    Analyzes human gait from MediaPipe pose landmarks.
    Tracks joint angles, detects gait cycles, computes symmetry & quality scores.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking state."""
        # Joint angles (current frame)
        self.angles = {
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
            "left_ankle": 0.0,
            "right_ankle": 0.0,
        }

        # Angle status classification
        self.angle_status = {k: "normal" for k in self.angles}

        # Gait cycle tracking
        self.step_count = 0
        self.cadence = 0.0  # steps per minute
        self._step_timestamps = deque(maxlen=50)
        self._prev_left_heel_y = None
        self._prev_right_heel_y = None
        self._left_heel_rising = False
        self._right_heel_rising = False

        # Symmetry
        self.symmetry_scores = {"hip": 100.0, "knee": 100.0, "ankle": 100.0}
        self.overall_symmetry = 100.0

        # Quality score
        self.gait_quality = 100.0

        # Abnormality flags
        self.abnormalities = []

        # History for smoothing
        self._angle_history = {k: deque(maxlen=10) for k in self.angles}

        # Timing
        self._start_time = time.time()

    def analyze(self, landmarks, frame_w: int, frame_h: int):
        """
        Analyze a single frame's worth of pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks list
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
        """
        if landmarks is None:
            return

        lm = landmarks.landmark

        # ── Extract key joint positions ──
        coords = {}
        indices = {
            "left_shoulder": LEFT_SHOULDER,
            "left_hip": LEFT_HIP,
            "left_knee": LEFT_KNEE,
            "left_ankle": LEFT_ANKLE,
            "left_heel": LEFT_HEEL,
            "left_foot": LEFT_FOOT_INDEX,
            "right_shoulder": RIGHT_SHOULDER,
            "right_hip": RIGHT_HIP,
            "right_knee": RIGHT_KNEE,
            "right_ankle": RIGHT_ANKLE,
            "right_heel": RIGHT_HEEL,
            "right_foot": RIGHT_FOOT_INDEX,
        }

        for name, idx in indices.items():
            coords[name] = get_landmark_coords(lm, idx, frame_w, frame_h)

        # ── Compute joint angles ──
        self._compute_angles(coords)

        # ── Detect gait cycle events ──
        self._detect_steps(coords, frame_h)

        # ── Compute symmetry ──
        self._compute_symmetry()

        # ── Compute overall quality ──
        self._compute_quality()

        # ── Flag abnormalities ──
        self._flag_abnormalities()

    def _compute_angles(self, coords):
        """Calculate all joint angles and classify them."""
        # Hip angle: shoulder → hip → knee
        self.angles["left_hip"] = calculate_angle(
            coords["left_shoulder"], coords["left_hip"], coords["left_knee"]
        )
        self.angles["right_hip"] = calculate_angle(
            coords["right_shoulder"], coords["right_hip"], coords["right_knee"]
        )

        # Knee angle: hip → knee → ankle
        self.angles["left_knee"] = calculate_angle(
            coords["left_hip"], coords["left_knee"], coords["left_ankle"]
        )
        self.angles["right_knee"] = calculate_angle(
            coords["right_hip"], coords["right_knee"], coords["right_ankle"]
        )

        # Ankle angle: knee → ankle → foot
        self.angles["left_ankle"] = calculate_angle(
            coords["left_knee"], coords["left_ankle"], coords["left_foot"]
        )
        self.angles["right_ankle"] = calculate_angle(
            coords["right_knee"], coords["right_ankle"], coords["right_foot"]
        )

        # Smooth angles and classify
        for key in self.angles:
            self._angle_history[key].append(self.angles[key])
            smoothed = np.mean(self._angle_history[key])
            self.angles[key] = round(smoothed, 1)

            # Classify based on joint type
            joint = key.split("_", 1)[1]  # "hip", "knee", or "ankle"
            self.angle_status[key] = classify_angle(self.angles[key], joint)

    def _detect_steps(self, coords, frame_h):
        """Detect heel-strike events to count steps."""
        # Normalize Y by frame height for consistent threshold
        left_heel_y = coords["left_heel"][1] / frame_h
        right_heel_y = coords["right_heel"][1] / frame_h

        now = time.time()

        # Left foot step detection
        if self._prev_left_heel_y is not None:
            dy = left_heel_y - self._prev_left_heel_y
            # Heel moving down (Y increases in image coords) = heel strike
            if dy > HEEL_STRIKE_THRESHOLD and self._left_heel_rising:
                self.step_count += 1
                self._step_timestamps.append(now)
                self._left_heel_rising = False
            elif dy < -HEEL_STRIKE_THRESHOLD:
                self._left_heel_rising = True

        # Right foot step detection
        if self._prev_right_heel_y is not None:
            dy = right_heel_y - self._prev_right_heel_y
            if dy > HEEL_STRIKE_THRESHOLD and self._right_heel_rising:
                self.step_count += 1
                self._step_timestamps.append(now)
                self._right_heel_rising = False
            elif dy < -HEEL_STRIKE_THRESHOLD:
                self._right_heel_rising = True

        self._prev_left_heel_y = left_heel_y
        self._prev_right_heel_y = right_heel_y

        # Compute cadence (steps per minute) from recent steps
        if len(self._step_timestamps) >= 2:
            window = [t for t in self._step_timestamps if now - t < 30]
            if len(window) >= 2:
                duration = window[-1] - window[0]
                if duration > 0:
                    self.cadence = round((len(window) - 1) / duration * 60, 1)

    def _compute_symmetry(self):
        """Compute left/right symmetry for each joint group."""
        for joint in ["hip", "knee", "ankle"]:
            left = self.angles.get(f"left_{joint}", 0)
            right = self.angles.get(f"right_{joint}", 0)
            diff = abs(left - right)
            # Score: 100 when perfect, decreasing with difference
            max_diff = 45.0  # max expected difference
            score = max(0, 100 - (diff / max_diff) * 100)
            self.symmetry_scores[joint] = round(score, 1)

        self.overall_symmetry = round(
            np.mean(list(self.symmetry_scores.values())), 1
        )

    def _compute_quality(self):
        """
        Compute overall gait quality score (0-100).
        Factors: joint angle normality + symmetry.
        """
        # Angle normality score
        status_scores = {"normal": 100, "warning": 60, "abnormal": 20}
        angle_scores = [status_scores[s] for s in self.angle_status.values()]
        avg_angle_score = np.mean(angle_scores)

        # Combine: 60% angle normality + 40% symmetry
        self.gait_quality = round(0.6 * avg_angle_score + 0.4 * self.overall_symmetry, 1)

    def _flag_abnormalities(self):
        """Generate human-readable abnormality descriptions."""
        flags = []

        # Check each joint angle
        for key, status in self.angle_status.items():
            if status == "abnormal":
                side = "Left" if "left" in key else "Right"
                joint = key.split("_", 1)[1].capitalize()
                angle = self.angles[key]
                r = NORMAL_RANGES.get(key.split("_", 1)[1], {})
                range_str = f"{r.get('min', '?')}°-{r.get('max', '?')}°"
                flags.append(f"{side} {joint}: {angle}° (normal: {range_str})")
            elif status == "warning":
                side = "Left" if "left" in key else "Right"
                joint = key.split("_", 1)[1].capitalize()
                angle = self.angles[key]
                flags.append(f"{side} {joint}: {angle}° — slight deviation")

        # Check symmetry
        for joint, score in self.symmetry_scores.items():
            if score < 70:
                left = self.angles.get(f"left_{joint}", 0)
                right = self.angles.get(f"right_{joint}", 0)
                diff = abs(left - right)
                flags.append(f"{joint.capitalize()} asymmetry: {diff:.1f}° difference")

        self.abnormalities = flags

    @property
    def elapsed_time(self) -> float:
        """Seconds since analysis started."""
        return time.time() - self._start_time

    @property
    def elapsed_str(self) -> str:
        """Formatted elapsed time string."""
        t = int(self.elapsed_time)
        m, s = divmod(t, 60)
        return f"{m:02d}:{s:02d}"

    def get_summary(self) -> dict:
        """Return a dictionary of all current analysis data."""
        return {
            "angles": dict(self.angles),
            "angle_status": dict(self.angle_status),
            "step_count": self.step_count,
            "cadence": self.cadence,
            "symmetry": dict(self.symmetry_scores),
            "overall_symmetry": self.overall_symmetry,
            "gait_quality": self.gait_quality,
            "abnormalities": list(self.abnormalities),
            "elapsed": self.elapsed_str,
        }
