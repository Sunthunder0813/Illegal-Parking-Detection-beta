import cv2
import threading
import time
import os
import datetime
import numpy as np
import logging
from flask import Flask, Response, render_template_string
import firebase_admin
from firebase_admin import credentials, db

from app_detect import detect
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

# --- TRACKER LOGIC ---
class SimpleIoUTracker:
    def __init__(self):
        self.prev_boxes = {} # ID: box
        self.next_id = 0

    def get_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, current_boxes):
        updated_boxes = {}
        for cur_box in current_boxes:
            best_id = None; max_iou = 0.3 # Threshold
            for tid, prev_box in self.prev_boxes.items():
                iou = self.get_iou(cur_box, prev_box)
                if iou > max_iou:
                    max_iou = iou; best_id = tid
            
            if best_id is not None:
                updated_boxes[best_id] = cur_box
                del self.prev_boxes[best_id]
            else:
                updated_boxes[self.next_id] = cur_box
                self.next_id += 1
        
        self.prev_boxes = updated_boxes
        return updated_boxes

# --- APP LOGIC ---
ZONE_CAM1 = np.array([[100, 300], [1100, 300], [1100, 700], [100, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [1000, 200], [1000, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
        except Exception: self.ref = None

    def report_violation(self, cam_name, frame):
        if self.ref:
            threading.Thread(target=self._upload, args=(cam_name, frame.copy()), daemon=True).start()

    def _upload(self, cam_name, frame):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{cam_name}_{ts}.jpg"
        cv2.imwrite(os.path.join(config.SAVE_DIR, fname), frame)
        self.ref.push({"timestamp": ts, "camera": cam_name, "status": "Illegal Parking"})

db_client = FirebaseHandler()

class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": SimpleIoUTracker(), "Camera_2": SimpleIoUTracker()}
        self.timers = {} # (cam, id): start_time
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}

    def process(self, cam_name, raw_boxes, frame):
        h, w = frame.shape[:2]
        cv2.polylines(frame, [self.zones[cam_name]], True, (0, 255, 0), 2)
        
        # Scale & Track
        pixel_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in raw_boxes]
        tracked = self.trackers[cam_name].update(pixel_boxes)
        
        now = time.time()
        for tid, box in tracked.items():
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            if cv2.pointPolygonTest(self.zones[cam_name], center, False) >= 0:
                key = (cam_name, tid)
                if key not in self.timers: self.timers[key] = now
                
                duration = now - self.timers[key]
                color = (0, 0, 255) if duration > config.VIOLATION_TIME_THRESHOLD else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {tid}: {int(duration)}s", (x1, y1-10), 0, 0.6, color, 2)
                
                if duration > config.VIOLATION_TIME_THRESHOLD:
                    db_client.report_violation(cam_name, frame)
            else:
                self.timers.pop((cam_name, tid), None)

monitor = ParkingMonitor()

class Stream:
    def __init__(self, url, name):
        self.cap = cv2.VideoCapture(url)
        self.name = name; self.frame = None
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret: self.frame = f
            else: time.sleep(1)

cam1 = Stream(config.CAM1_URL, "Camera_1")
cam2 = Stream(config.CAM2_URL, "Camera_2")
app = Flask(__name__)

def gen():
    while True:
        cams = []
        if cam1.frame is not None: cams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: cams.append(("Camera_2", cam2.frame.copy()))
        if not cams: time.sleep(0.1); continue

        results = detect([f for n, f in cams])
        imgs = []
        for i, res in enumerate(results):
            name, frame = cams[i]
            monitor.process(name, res.boxes.xyxy, frame)
            imgs.append(cv2.resize(frame, (640, 480)))

        combined = cv2.hconcat(imgs if len(imgs)==2 else [imgs[0], np.zeros_like(imgs[0])])
        _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return "<h1>Parking Monitor</h1><img src='/video_feed' width='100%'>"

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
