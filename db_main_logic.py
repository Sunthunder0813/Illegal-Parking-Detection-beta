import cv2
import threading
import time
import os
import json
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, render_template, jsonify, request
import datetime
import argparse
import sys
import contextlib
import io

from detect import NAMES, CLASSES, detect  # Import detect function as well
from config import CAM1_URL, CAM2_URL, SAVE_DIR, DATABASE_URL, FIREBASE_KEY_PATH, UPLOAD_COOLDOWN

# =============================
# CONFIG & FIREBASE
# =============================
os.makedirs(SAVE_DIR, exist_ok=True)

class FirebaseHandler:
    def __init__(self):
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
            self.ref = db.reference('violations_history')
            print("Connected to Firebase Realtime Database")
        except Exception as e:
            print(f"Firebase Error: {e}")

    def report_violation(self, cam_names, total_count, merged_frame):
        """Saves merged image locally and pushes metadata to Firebase."""
        if total_count > 0:
            threading.Thread(target=self._process_local_save, args=(cam_names, total_count, merged_frame.copy()), daemon=True).start()

    def _process_local_save(self, cam_names, total_count, merged_frame):
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{'_'.join(cam_names)}_{timestamp_str}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, merged_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            self.ref.push({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cameras": cam_names,
                "vehicle_count": total_count,
                "local_file": filename,
                "status": "Illegal Parking Logged Locally"
            })
            print(f"Logged: {cam_names} | Image saved to {filepath}")
        except Exception as e:
            print(f"Local Save Error: {e}")

db_client = FirebaseHandler()

# =============================
# CAMERA CLASSES
# =============================
class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.status, self.frame = False, None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            ret, frame = self.cap.read()
            if ret:
                self.status, self.frame = True, frame
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

class WebcamStream:
    def __init__(self, cam_index):
        self.cam_index = cam_index
        self.cap = self._safe_videocapture(cam_index)
        self.status, self.frame = False, None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _safe_videocapture(self, index):
        with contextlib.redirect_stderr(io.StringIO()):
            return cv2.VideoCapture(index)

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(2)
                self.cap = self._safe_videocapture(self.cam_index)
                continue
            ret, frame = self.cap.read()
            if ret:
                self.status, self.frame = True, frame
            else:
                self.frame = None
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# =============================
# ARGUMENTS
# =============================
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true', help='Use local webcams instead of IP cameras')
args, unknown = parser.parse_known_args()

# =============================
# CAMERA INITIALIZATION
# =============================
if args.webcam:
    # Detect available webcams automatically
    available_cams = []
    for i in range(5):  # Check first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam index {i} is available")
            available_cams.append(i)
        cap.release()

    if len(available_cams) == 0:
        raise Exception("No webcams found on this system.")

    cam1 = WebcamStream(available_cams[0])
    cam2 = WebcamStream(available_cams[1]) if len(available_cams) > 1 else None
    print(f"Using webcams: {available_cams}")
else:
    cam1 = RTSPStream(CAM1_URL)
    cam2 = RTSPStream(CAM2_URL)
    print("Using IP cameras (RTSP streams)")

# =============================
# SETTINGS FILE
# =============================
SETTINGS_FILE = "settings.json"

def load_settings():
    """Load settings from JSON file or return defaults"""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    else:
        from config import VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL
        return {
            "VIOLATION_TIME_THRESHOLD": VIOLATION_TIME_THRESHOLD,
            "REPEAT_CAPTURE_INTERVAL": REPEAT_CAPTURE_INTERVAL
        }

def save_settings(settings):
    """Save settings to JSON file"""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# Load settings at startup
current_settings = load_settings()
VIOLATION_TIME_THRESHOLD = current_settings["VIOLATION_TIME_THRESHOLD"]
REPEAT_CAPTURE_INTERVAL = current_settings["REPEAT_CAPTURE_INTERVAL"]

# =============================
# FLASK APP
# =============================
app = Flask(__name__)

def gen_frames():
    last_upload_time = 0

    while True:
        frames = []
        names = []

        f1 = cam1.read() if cam1 else None
        f2 = cam2.read() if cam2 else None

        if f1 is not None:
            frames.append(f1)
            names.append("Camera_1")
        if f2 is not None:
            frames.append(f2)
            names.append("Camera_2")

        if not frames:
            time.sleep(0.05)
            continue

        results = detect(frames)  # Should return a list of result objects, one per frame

        # Prepare merged snapshot for violation
        plotted_frames = [res.plot() for res in results]
        if len(plotted_frames) == 2:
            merged_snapshot = cv2.hconcat(plotted_frames)
        elif len(plotted_frames) == 1:
            merged_snapshot = plotted_frames[0]
        else:
            merged_snapshot = None

        current_time = time.time()
        total_count = sum(len(res.boxes) for res in results)
        if merged_snapshot is not None and total_count > 0 and current_time - last_upload_time > UPLOAD_COOLDOWN:
            db_client.report_violation(names, total_count, merged_snapshot)
            last_upload_time = current_time

        if merged_snapshot is not None:
            _, buffer = cv2.imencode('.jpg', merged_snapshot)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(current_settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global current_settings, VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL
    data = request.get_json()
    current_settings = data
    save_settings(data)
    VIOLATION_TIME_THRESHOLD = data["VIOLATION_TIME_THRESHOLD"]
    REPEAT_CAPTURE_INTERVAL = data["REPEAT_CAPTURE_INTERVAL"]
    return jsonify({"success": True})

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    sys.argv = [sys.argv[0]]  # Remove custom args for Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)
