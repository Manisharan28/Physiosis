"""
Physiosis — Live overlay rendering for gait analysis visualization.
Draws skeleton, metrics panel, and abnormality warnings on video frames.
"""
import cv2
import numpy as np
import mediapipe as mp

from utils import status_color

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ──────────────────────────────────────────────
# Custom drawing spec for skeleton
# ──────────────────────────────────────────────
SKELETON_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 200), thickness=2, circle_radius=3
)
CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=(0, 200, 180), thickness=2
)

# Panel dimensions
PANEL_WIDTH = 340
PANEL_BG = (20, 20, 30)           # Dark background
PANEL_HEADER_BG = (40, 30, 60)    # Purple-ish header
ACCENT_CYAN = (255, 200, 0)       # Cyan accent (BGR)
ACCENT_MAGENTA = (200, 50, 255)   # Magenta accent
TEXT_PRIMARY = (240, 240, 240)     # White text
TEXT_SECONDARY = (160, 160, 180)   # Gray text
TEXT_DIM = (100, 100, 120)         # Dimmer text


def draw_skeleton(frame, results):
    """
    Draw MediaPipe pose skeleton on the frame with custom styling.
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=SKELETON_SPEC,
            connection_drawing_spec=CONNECTION_SPEC,
        )


def create_analysis_panel(analyzer, frame_h):
    """
    Create a side panel image showing real-time analysis metrics.
    
    Args:
        analyzer: GaitAnalyzer instance with current data
        frame_h: Height of the video frame (panel matches this height)
    
    Returns:
        np.ndarray: The panel image (frame_h x PANEL_WIDTH x 3)
    """
    panel = np.full((frame_h, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    y = 0

    # ── Header ──
    y = _draw_header(panel, y, analyzer)

    # ── Gait Quality Score ──
    y = _draw_quality_section(panel, y, analyzer)

    # ── Joint Angles ──
    y = _draw_angles_section(panel, y, analyzer)

    # ── Gait Metrics ──
    y = _draw_metrics_section(panel, y, analyzer)

    # ── Symmetry ──
    y = _draw_symmetry_section(panel, y, analyzer)

    # ── Abnormalities ──
    y = _draw_abnormalities_section(panel, y, analyzer, frame_h)

    return panel


def _draw_header(panel, y, analyzer):
    """Draw the panel header with title and elapsed time."""
    h = 55
    cv2.rectangle(panel, (0, y), (PANEL_WIDTH, y + h), PANEL_HEADER_BG, -1)

    # Title
    cv2.putText(
        panel, "PHYSIOSIS",
        (15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_CYAN, 2
    )
    # Subtitle
    cv2.putText(
        panel, "Gait Analysis",
        (15, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )
    # Timer
    cv2.putText(
        panel, analyzer.elapsed_str,
        (PANEL_WIDTH - 75, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1
    )

    # Separator line
    y += h
    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), ACCENT_CYAN, 1)
    return y + 5


def _draw_quality_section(panel, y, analyzer):
    """Draw the gait quality score with a colored arc/bar."""
    y += 10
    cv2.putText(
        panel, "GAIT QUALITY", (15, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )

    score = analyzer.gait_quality
    if score >= 80:
        color = (0, 220, 0)    # Green
        label = "Good"
    elif score >= 60:
        color = (0, 200, 255)  # Yellow
        label = "Fair"
    elif score >= 40:
        color = (0, 140, 255)  # Orange
        label = "Poor"
    else:
        color = (0, 0, 255)    # Red
        label = "Critical"

    # Score number
    cv2.putText(
        panel, f"{score:.0f}",
        (15, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
    )
    cv2.putText(
        panel, f"/ 100  {label}",
        (80, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1
    )

    # Progress bar
    bar_x, bar_y = 15, y + 55
    bar_w = PANEL_WIDTH - 30
    bar_h = 8
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
    fill_w = int(bar_w * score / 100)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    y += 72
    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (50, 50, 60), 1)
    return y + 5


def _draw_angles_section(panel, y, analyzer):
    """Draw joint angles with color-coded status indicators."""
    y += 5
    cv2.putText(
        panel, "JOINT ANGLES", (15, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )
    y += 22

    joints = [
        ("Hip", "left_hip", "right_hip"),
        ("Knee", "left_knee", "right_knee"),
        ("Ankle", "left_ankle", "right_ankle"),
    ]

    for label, left_key, right_key in joints:
        y += 22
        # Joint label
        cv2.putText(
            panel, label, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_PRIMARY, 1
        )

        # Left value
        l_angle = analyzer.angles[left_key]
        l_status = analyzer.angle_status[left_key]
        l_color = status_color(l_status)
        cv2.putText(
            panel, f"L:{l_angle:.0f}", (100, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, l_color, 1
        )
        # Status dot
        cv2.circle(panel, (90, y - 4), 4, l_color, -1)

        # Right value
        r_angle = analyzer.angles[right_key]
        r_status = analyzer.angle_status[right_key]
        r_color = status_color(r_status)
        cv2.putText(
            panel, f"R:{r_angle:.0f}", (210, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, r_color, 1
        )
        cv2.circle(panel, (200, y - 4), 4, r_color, -1)

    y += 15
    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (50, 50, 60), 1)
    return y + 5


def _draw_metrics_section(panel, y, analyzer):
    """Draw step count and cadence."""
    y += 5
    cv2.putText(
        panel, "GAIT METRICS", (15, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )
    y += 30

    # Steps
    cv2.putText(
        panel, "Steps:", (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1
    )
    cv2.putText(
        panel, str(analyzer.step_count), (120, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_CYAN, 1
    )

    # Cadence
    cv2.putText(
        panel, "Cadence:", (170, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1
    )
    cv2.putText(
        panel, f"{analyzer.cadence:.0f}/min", (255, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_CYAN, 1
    )

    y += 15
    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (50, 50, 60), 1)
    return y + 5


def _draw_symmetry_section(panel, y, analyzer):
    """Draw symmetry scores per joint group."""
    y += 5
    cv2.putText(
        panel, "SYMMETRY", (15, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )
    y += 25

    for joint in ["hip", "knee", "ankle"]:
        y += 20
        score = analyzer.symmetry_scores[joint]
        color = (0, 220, 0) if score >= 85 else (0, 200, 255) if score >= 70 else (0, 0, 255)

        cv2.putText(
            panel, f"{joint.capitalize()}", (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_PRIMARY, 1
        )

        # Mini bar
        bar_x = 100
        bar_w = 150
        bar_h = 10
        cv2.rectangle(panel, (bar_x, y - 8), (bar_x + bar_w, y - 8 + bar_h), (50, 50, 60), -1)
        fill = int(bar_w * score / 100)
        cv2.rectangle(panel, (bar_x, y - 8), (bar_x + fill, y - 8 + bar_h), color, -1)

        # Percentage
        cv2.putText(
            panel, f"{score:.0f}%", (bar_x + bar_w + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )

    y += 15
    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (50, 50, 60), 1)
    return y + 5


def _draw_abnormalities_section(panel, y, analyzer, frame_h):
    """Draw flagged abnormalities with warning icons."""
    y += 5
    cv2.putText(
        panel, "FINDINGS", (15, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1
    )
    y += 25

    if not analyzer.abnormalities:
        cv2.putText(
            panel, "No issues detected", (20, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1
        )
        # Checkmark
        cv2.putText(
            panel, "[OK]", (PANEL_WIDTH - 55, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1
        )
        return y + 20

    for flag in analyzer.abnormalities:
        if y + 20 > frame_h - 10:
            cv2.putText(
                panel, f"... +{len(analyzer.abnormalities)} more",
                (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1
            )
            break

        # Warning indicator
        is_severe = "abnormal" in flag.lower() or "asymmetry" in flag.lower()
        color = (0, 0, 255) if is_severe else (0, 200, 255)
        prefix = "[!]" if is_severe else "[~]"

        cv2.putText(
            panel, prefix, (15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )

        # Truncate long text
        text = flag if len(flag) < 35 else flag[:32] + "..."
        cv2.putText(
            panel, text, (40, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1
        )
        y += 18

    return y


def draw_abnormality_banner(frame, abnormalities):
    """
    Draw a warning banner at the top of the frame when abnormalities are found.
    """
    if not abnormalities:
        return

    h, w = frame.shape[:2]

    # Semi-transparent red overlay at top
    overlay = frame.copy()
    banner_h = 35
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Warning text
    count = len(abnormalities)
    text = f"WARNING: {count} gait abnormalit{'y' if count == 1 else 'ies'} detected"
    cv2.putText(
        frame, text,
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )


def draw_controls_hint(frame):
    """Draw keyboard controls hint at the bottom of the frame."""
    h, w = frame.shape[:2]
    hints = "Q: Quit | R: Reset | S: Screenshot | P: Pause"
    cv2.putText(
        frame, hints,
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1
    )


def compose_frame(frame, panel):
    """
    Combine the video frame and analysis panel side by side.
    
    Returns:
        np.ndarray: Composed output frame
    """
    frame_h = frame.shape[0]
    panel_h = panel.shape[0]

    # Ensure panel matches frame height
    if panel_h != frame_h:
        panel = cv2.resize(panel, (PANEL_WIDTH, frame_h))

    return np.hstack([frame, panel])
