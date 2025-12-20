import cv2
import threading
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, render_template
import datetime
import argparse
import sys
import contextlib
import io

from detect import NAMES, CLASSES, detect  # Import detect function as well

# =============================
# CONFIG & FIREBASE
# =============================
DATABASE_URL = "https://illegal-parking-detectio-a8aae-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_KEY_PATH = "illegal-parking-detectio-a8aae-firebase-adminsdk-fbsvc-7181a051dc.json"

SAVE_DIR = "static/violations"
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

    def report_violation(self, cam_name, count, frame):
        if count > 0:
            threading.Thread(target=self._process_local_save, args=(cam_name, count, frame.copy()), daemon=True).start()

    def _process_local_save(self, cam_name, count, frame):
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cam_name}_{timestamp_str}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            self.ref.push({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_name,
                "vehicle_count": count,
                "local_file": filename,
                "status": "Illegal Parking Logged Locally"
            })
            print(f"Logged: {cam_name} | Image saved to {filepath}")
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

CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.113:554/stream"

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
# FLASK APP
# =============================
app = Flask(__name__)

def gen_frames():
    UPLOAD_COOLDOWN = 30
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

        # Use detect.detect instead of model.track
        results = detect(frames)  # Should return a list of result objects, one per frame

        current_time = time.time()
        for i, res in enumerate(results):
            count = len(res.boxes)
            if count > 0 and current_time - last_upload_time > UPLOAD_COOLDOWN:
                db_client.report_violation(names[i], count, res.plot())
                last_upload_time = current_time

        combined = cv2.hconcat([res.plot() for res in results])
        _, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    sys.argv = [sys.argv[0]]  # Remove custom args for Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)
