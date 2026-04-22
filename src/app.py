"""
Physiosis — Flask Web Service
Serves the dark mode web frontend and streams MJPEG video from the gait analyzer.
"""
import cv2
import time
import argparse
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp

from video_handler import VideoHandler
from gait_analyzer import GaitAnalyzer
from overlay import (
    draw_skeleton,
    create_analysis_panel,
    draw_abnormality_banner,
    compose_frame
)

app = Flask(__name__)

# Global state for the video processing logic
analyzer = GaitAnalyzer()
video = None
pose = None
is_analyzing = False
frame_w, frame_h = 960, 640

def init_vision(source=0):
    """Initialize OpenCV and MediaPipe components."""
    global video, pose, is_analyzing, analyzer
    
    if video is not None:
        video.release()
    
    try:
        video = VideoHandler(source)
    except Exception as e:
        print(f"Error opening video source: {e}")
        return False
        
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    is_analyzing = True
    analyzer.reset()
    return True

def generate_frames():
    """Generator yielding JPEG frames for the MJPEG stream."""
    global video, pose, is_analyzing, analyzer, frame_w, frame_h
    
    # Delay for reading video files to simulate realtime
    delay_s = 1.0 / video.fps if not video.is_webcam else 0.0
    
    while is_analyzing:
        t_start = time.time()
        
        ret, frame = video.read_frame()
        if not ret:
            if not video.is_webcam:
                break # video ended
            continue
            
        frame = cv2.resize(frame, (frame_w, frame_h))
        cur_h, cur_w = frame.shape[:2]
        
        # Pose estimation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Analysis
        if results.pose_landmarks:
            analyzer.analyze(results.pose_landmarks, cur_w, cur_h)
            draw_skeleton(frame, results)
            
        # Draw Overlays
        draw_abnormality_banner(frame, analyzer.abnormalities)
        panel = create_analysis_panel(analyzer, cur_h)
        composed = compose_frame(frame, panel)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', composed)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Throttle processing loop if it's a video file
        elapsed = time.time() - t_start
        if not video.is_webcam and elapsed < delay_s:
            time.sleep(delay_s - elapsed)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main web dashboard dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route handling the MJPEG streaming."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Return JSON indicating if analyzer is currently active."""
    return jsonify({"active": is_analyzing})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Physiosis Web Server")
    parser.add_argument("--source", default="0", help="Camera index or video file map")
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    print("Initializing Vision System...")
    if init_vision(source):
        print("Starting Flask Web Server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("Failed to initialize vision system. Check your camera or video file.")
