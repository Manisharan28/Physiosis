import cv2
import base64
import math
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="Physiosis WebSocket Engine")

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Video Mapping ─────────────────────────────────────────────────────────────
EXERCISE_VIDEOS = {
    "Forward Head Posture": "https://www.youtube.com/watch?v=LT_dFRnmdGs",
    "Neck Tilt":            "https://www.youtube.com/watch?v=wQylqaCl8Zo",
    "Shoulder Imbalance":   "https://www.youtube.com/watch?v=DFRRJYPQCCw",
    "Pelvic Tilt":          "https://www.youtube.com/watch?v=R-aA2FuRBFk",
    "Elbow Hyperextension": "https://www.youtube.com/watch?v=zmzHEBSvloI",
    "Knee Valgus":          "https://www.youtube.com/watch?v=o1I1eiMnd1I",
    "Knee Hyperextension":  "https://www.youtube.com/watch?v=o1I1eiMnd1I",
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def clamp(val: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, val)))


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash=10, gap=6):
    """Draw a dashed line segment between two pixel-coordinate tuples."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    x, y, drawn, drawing = float(x1), float(y1), 0.0, True
    while drawn < dist:
        step = min(dash if drawing else gap, dist - drawn)
        ex, ey = x + dx * step, y + dy * step
        if drawing:
            cv2.line(img, (int(x), int(y)), (int(ex), int(ey)), color, thickness)
        x, y, drawn = ex, ey, drawn + step
        drawing = not drawing


def get_360_angle(a, b, c, is_left=True):
    """0-360 deg angle at joint b. Straight=180, flexion<180, hyperextension>180."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_a = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8), -1, 1
    )
    angle = np.degrees(np.arccos(cos_a))
    cross = ba[0] * bc[1] - ba[1] * bc[0]
    if is_left and cross < 0:
        angle = 360 - angle
    elif not is_left and cross > 0:
        angle = 360 - angle
    return angle


VISIBILITY_THRESHOLD = 0.6
GHOST = (0, 220, 110)   # bright green  – ideal target line
GUIDE = (0, 180, 255)   # amber-blue    – shift connector


# ── Core analyser ─────────────────────────────────────────────────────────────

def analyze_biomechanics(landmarks, frame):
    """
    Evaluate 6 biomechanical rules.
    Returns list of dicts: {issue, status, tip, severity}.
    Ghost-skeleton corrections are drawn directly onto `frame`.
    """
    diagnostics = []
    h, w = frame.shape[:2]

    def get_pt(idx, return_z=False):
        lm = landmarks[idx]
        if return_z:
            return [lm.x, lm.y, lm.z], lm.visibility
        return [lm.x, lm.y], lm.visibility

    def px(pt):
        return (int(pt[0] * w), int(pt[1] * h))

    def vis(*vs):
        return all(v >= VISIBILITY_THRESHOLD for v in vs)

    # ── Rule 1 · Forward Head Posture ─────────────────────────────────────────
    l_ear,      v_le = get_pt(mp_pose.PoseLandmark.LEFT_EAR.value, return_z=True)
    l_shoulder, v_ls = get_pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value, return_z=True)
    r_shoulder, v_rs = get_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value, return_z=True)

    if vis(v_le, v_ls, v_rs):
        width = abs(l_shoulder[0] - r_shoulder[0])
        is_frontal = width > 0.1
        
        if is_frontal:
            # 3D robust anterior head displacement for frontal view
            mid_shoulder = (np.array(l_shoulder) + np.array(r_shoulder)) / 2.0
            shoulder_vec = np.array(l_shoulder) - np.array(r_shoulder)
            forward_vec = np.cross([0, 1, 0], shoulder_vec)
            norm = np.linalg.norm(forward_vec)
            forward_vec = forward_vec / norm if norm > 1e-6 else np.array([0, 0, -1])
            head_vec = np.array(l_ear) - mid_shoulder
            fhp = float(np.dot(head_vec, forward_vec))
        else:
            # Fallback to standard 2D lateral offset for pure side profiles
            fhp = abs(l_ear[0] - l_shoulder[0])

        fhp_val = max(0.0, fhp)
        cm  = fhp_val * 60
        if fhp_val > 0.08:
            sev = clamp((fhp_val - 0.08) / 0.17 * 100, 0, 100)
            tip = (f"Head is ~{cm:.1f} cm forward of shoulder. "
                   f"Tuck chin in and stack your ears directly over your shoulders.")
            diagnostics.append({"issue": "Forward Head Posture", "status": "Bad", "tip": tip, "severity": sev})
            ideal = (int(l_shoulder[0] * w), int(l_ear[1] * h))
            draw_dashed_line(frame, px(l_ear), ideal, GHOST, 2)
            draw_dashed_line(frame, ideal, px(l_shoulder), GUIDE, 1)
        else:
            diagnostics.append({"issue": "Forward Head Posture", "status": "Good",
                                 "tip": f"Great neck alignment — offset is only ~{cm:.1f} cm.", "severity": 0})
    else:
        diagnostics.append({"issue": "Forward Head Posture", "status": "WAITING",
                             "tip": "Ensure your head and shoulders are fully visible.", "severity": 0})

    # ── Rule 2 · Neck Tilt ────────────────────────────────────────────────────
    nose,       v_n  = get_pt(mp_pose.PoseLandmark.NOSE.value)
    r_shoulder, v_rs = get_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

    if vis(v_n, v_ls, v_rs):
        mid  = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        vert = [mid[0], mid[1] - 0.5]
        na   = math.degrees(math.atan2(nose[1] - mid[1], nose[0] - mid[0]))
        va   = math.degrees(math.atan2(vert[1] - mid[1], vert[0] - mid[0]))
        tilt = abs(na - va)
        if tilt > 180:
            tilt = 360 - tilt
        if tilt > 10:
            sev = clamp((tilt - 10) / 35 * 100, 0, 100)
            tip = (f"Neck tilted ~{tilt:.1f} to the side. "
                   f"Level your head — try reducing tilt by ~{tilt - 10:.1f}.")
            diagnostics.append({"issue": "Neck Tilt", "status": "Bad", "tip": tip, "severity": sev})
            ideal_nose = (int(mid[0] * w), int(nose[1] * h))
            draw_dashed_line(frame, px(nose), ideal_nose, GHOST, 2)
        else:
            diagnostics.append({"issue": "Neck Tilt", "status": "Good",
                                 "tip": f"Neck nicely leveled — tilt is just {tilt:.1f}.", "severity": 0})
    else:
        diagnostics.append({"issue": "Neck Tilt", "status": "WAITING",
                             "tip": "Ensure your face and both shoulders are clearly visible.", "severity": 0})

    # ── Rule 3 · Shoulder Imbalance ───────────────────────────────────────────
    if vis(v_ls, v_rs):
        diff = abs(l_shoulder[1] - r_shoulder[1])
        cm   = diff * 60
        if diff > 0.04:
            sev    = clamp((diff - 0.04) / 0.11 * 100, 0, 100)
            higher = "Left" if l_shoulder[1] < r_shoulder[1] else "Right"
            tip    = (f"{higher} shoulder is ~{cm:.1f} cm higher. "
                      f"Consciously relax the raised side and breathe out slowly.")
            diagnostics.append({"issue": "Shoulder Imbalance", "status": "Bad", "tip": tip, "severity": sev})
            mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
            draw_dashed_line(frame,
                             (int(l_shoulder[0] * w), int(mid_y * h)),
                             (int(r_shoulder[0] * w), int(mid_y * h)), GHOST, 2)
        else:
            diagnostics.append({"issue": "Shoulder Imbalance", "status": "Good",
                                 "tip": f"Shoulders balanced — diff only {cm:.1f} cm.", "severity": 0})
    else:
        diagnostics.append({"issue": "Shoulder Imbalance", "status": "WAITING",
                             "tip": "Ensure both shoulders are clearly visible.", "severity": 0})

    # ── Rule 4 · Pelvic Tilt ─────────────────────────────────────────────────
    l_hip, v_lh = get_pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    r_hip, v_rh = get_pt(mp_pose.PoseLandmark.RIGHT_HIP.value)

    if vis(v_lh, v_rh):
        diff = abs(l_hip[1] - r_hip[1])
        cm   = diff * 60
        if diff > 0.04:
            sev    = clamp((diff - 0.04) / 0.11 * 100, 0, 100)
            higher = "Left" if l_hip[1] < r_hip[1] else "Right"
            tip    = (f"Hips off by ~{cm:.1f} cm ({higher} hip higher). "
                      f"Check for leg-length discrepancy or uneven weight distribution.")
            diagnostics.append({"issue": "Pelvic Tilt", "status": "Bad", "tip": tip, "severity": sev})
            mid_y = (l_hip[1] + r_hip[1]) / 2
            draw_dashed_line(frame,
                             (int(l_hip[0] * w), int(mid_y * h)),
                             (int(r_hip[0] * w), int(mid_y * h)), GHOST, 2)
        else:
            diagnostics.append({"issue": "Pelvic Tilt", "status": "Good",
                                 "tip": f"Hips balanced — diff only {cm:.1f} cm.", "severity": 0})
    else:
        diagnostics.append({"issue": "Pelvic Tilt", "status": "WAITING",
                             "tip": "Ensure both hips are clearly visible.", "severity": 0})

    # ── Rule 5 · Elbow Hyperextension ────────────────────────────────────────
    l_elbow, v_le2 = get_pt(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    l_wrist, v_lw  = get_pt(mp_pose.PoseLandmark.LEFT_WRIST.value)
    r_elbow, v_re  = get_pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    r_wrist, v_rw  = get_pt(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    if vis(v_ls, v_le2, v_lw, v_rs, v_re, v_rw):
        la = get_360_angle(l_shoulder, l_elbow, l_wrist, is_left=True)
        ra = get_360_angle(r_shoulder, r_elbow, r_wrist, is_left=False)
        mx = max(la, ra)
        if mx > 185:
            sev  = clamp((mx - 185) / 25 * 100, 0, 100)
            side = "Left" if la > ra else "Right"
            tip  = (f"{side} elbow at {mx:.1f} — hyperextended by {mx - 180:.1f}. "
                    f"Soften by ~{mx - 175:.1f} to protect the joint.")
            diagnostics.append({"issue": "Elbow Hyperextension", "status": "Bad", "tip": tip, "severity": sev})
            if la > ra:
                ie = [(l_shoulder[0] + l_wrist[0]) / 2, (l_shoulder[1] + l_wrist[1]) / 2]
                draw_dashed_line(frame, px(l_shoulder), px(ie),      GHOST, 2)
                draw_dashed_line(frame, px(ie),          px(l_wrist), GHOST, 2)
            else:
                ie = [(r_shoulder[0] + r_wrist[0]) / 2, (r_shoulder[1] + r_wrist[1]) / 2]
                draw_dashed_line(frame, px(r_shoulder), px(ie),      GHOST, 2)
                draw_dashed_line(frame, px(ie),          px(r_wrist), GHOST, 2)
        else:
            diagnostics.append({"issue": "Elbow Extension", "status": "Good",
                                 "tip": f"Elbows within safe range. Max angle: {mx:.1f}.", "severity": 0})
    else:
        diagnostics.append({"issue": "Elbow Extension", "status": "WAITING",
                             "tip": "Ensure shoulders, elbows, and wrists are clearly visible.", "severity": 0})

    # ── Rule 6 · Knee Valgus & Hyperextension ────────────────────────────────
    l_knee, v_lk  = get_pt(mp_pose.PoseLandmark.LEFT_KNEE.value)
    l_ankle, v_la = get_pt(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    r_knee, v_rk  = get_pt(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle, v_ra = get_pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    if vis(v_lh, v_lk, v_la, v_rh, v_rk, v_ra):
        # Knees hinge back (anatomically opposite to elbows), so we invert the is_left 
        # flag for the cross product mapping to differentiate flexion from hyperextension
        lka = get_360_angle(l_hip, l_knee, l_ankle, is_left=False)
        rka = get_360_angle(r_hip, r_knee, r_ankle, is_left=True)

        if max(lka, rka) > 185:
            ba   = max(lka, rka)
            sev  = clamp((ba - 185) / 25 * 100, 0, 100)
            side = "Left" if lka > rka else "Right"
            tip  = (f"{side} knee at {ba:.1f} (hyperextended by {ba - 180:.1f}). "
                    f"Slightly bend knee by ~{ba - 175:.1f} to protect the joint.")
            diagnostics.append({"issue": "Knee Hyperextension", "status": "Bad", "tip": tip, "severity": sev})
            if lka > rka:
                draw_dashed_line(frame, px(l_hip), px(l_ankle), GHOST, 2)
            else:
                draw_dashed_line(frame, px(r_hip), px(r_ankle), GHOST, 2)

        elif min(lka, rka) < 165:
            ba   = min(lka, rka)
            sev  = clamp((165 - ba) / 35 * 100, 0, 100)
            side = "Left" if lka < rka else "Right"
            tip  = (f"{side} knee at {ba:.1f} (valgus/knock-knee). "
                    f"Push knee outward by ~{165 - ba:.1f} to align with hip and ankle.")
            diagnostics.append({"issue": "Knee Valgus", "status": "Bad", "tip": tip, "severity": sev})
            if lka < rka:
                draw_dashed_line(frame, px(l_hip), px(l_ankle), GHOST, 2)
            else:
                draw_dashed_line(frame, px(r_hip), px(r_ankle), GHOST, 2)
        else:
            diagnostics.append({"issue": "Knee Alignment", "status": "Good",
                                 "tip": f"Knees aligned well. L: {lka:.1f} · R: {rka:.1f}.", "severity": 0})
    else:
        diagnostics.append({"issue": "Knee Alignment", "status": "WAITING",
                             "tip": "Ensure hips, knees, and ankles are clearly visible.", "severity": 0})

    for d in diagnostics:
        d["video"] = EXERCISE_VIDEOS.get(d["issue"])
        
    return diagnostics


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to /ws/analyze")
    try:
        while True:
            data = await websocket.receive_text()
            _, encoded = data.split(",", 1) if data.startswith("data:image/") else (None, data)

            frame_bytes = base64.b64decode(encoded)
            np_arr      = np.frombuffer(frame_bytes, np.uint8)
            frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(rgb_frame)

            diagnostics = []
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117,  66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,  66, 230), thickness=2, circle_radius=2),
                )
                try:
                    diagnostics = analyze_biomechanics(results.pose_landmarks.landmark, frame)
                except Exception as e:
                    print(f"Biomechanics error: {e}")

            _, buf  = cv2.imencode('.jpg', frame)
            out_b64 = base64.b64encode(buf).decode('utf-8')
            await websocket.send_json({
                "image":       f"data:image/jpeg;base64,{out_b64}",
                "diagnostics": diagnostics,
            })

    except WebSocketDisconnect:
        print("Client disconnected from /ws/analyze")


@app.get("/")
def get_dashboard():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_dir, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
