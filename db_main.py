import cv2
import threading
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO
from flask import Flask, Response, render_template
import datetime
from detect import NAMES, CLASSES

# =============================
# CONFIG & FIREBASE SETUP
# =============================
DATABASE_URL = "https://illegal-parking-detectio-a8aae-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_KEY_PATH = "illegal-parking-detectio-a8aae-firebase-adminsdk-fbsvc-7181a051dc.json"

# Create a local folder to store images
# Using 'static' folder allows Flask to display these images on your webpage easily
SAVE_DIR = "static/violations"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class FirebaseHandler:
    def __init__(self):
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
            # Reference for detection logs
            self.ref = db.reference('violations_history')
            print("Connected to Firebase Realtime Database (Free Tier)")
        except Exception as e:
            print(f"Firebase Error: {e}")

    def report_violation(self, cam_name, count, frame):
        """Saves image locally and pushes metadata to Firebase."""
        if count > 0:
            threading.Thread(target=self._process_local_save, args=(cam_name, count, frame.copy()), daemon=True).start()

    def _process_local_save(self, cam_name, count, frame):
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cam_name}_{timestamp_str}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            # Save the image locally with 70% quality to save disk space
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            cv2.imwrite(filepath, frame, encode_param)

            # Push the record to Firebase Realtime Database
            self.ref.push({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_name,
                "vehicle_count": count,
                "local_file": filename, # Store the filename to reference later
                "status": "Illegal Parking Logged Locally"
            })
            print(f"Logged: {cam_name} | Image saved to {filepath}")
        except Exception as e:
            print(f"Local Save Error: {e}")

db_client = FirebaseHandler()

# =============================
# CAMERA & MODEL SETUP
# =============================
MODEL_PATH = "models/yolov8n.pt" 
CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.113:554/stream"

class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    def read(self): return self.frame
    def stop(self):
        self.stopped = True
        self.cap.release()

model = YOLO(MODEL_PATH)
cam1 = RTSPStream(CAM1_URL)
cam2 = RTSPStream(CAM2_URL)
app = Flask(__name__)

# =============================
# DETECTION & STREAMING
# =============================
def gen_frames():
    UPLOAD_COOLDOWN = 30.0 # Capture every 30 seconds if car is present
    last_upload_time = 0
    
    while True:
        f1, f2 = cam1.read(), cam2.read()
        if f1 is None or f2 is None: continue

        results = model.track(
            [f1, f2],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=CLASSES  # <-- Use configurable classes
        )

        current_time = time.time()
        if current_time - last_upload_time > UPLOAD_COOLDOWN:
            c1, c2 = len(results[0].boxes), len(results[1].boxes)
            
            if c1 > 0:
                db_client.report_violation("Camera_1", c1, results[0].plot())
                last_upload_time = current_time
            elif c2 > 0:
                db_client.report_violation("Camera_2", c2, results[1].plot())
                last_upload_time = current_time

        combined = cv2.hconcat([results[0].plot(), results[1].plot()])
        _, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)