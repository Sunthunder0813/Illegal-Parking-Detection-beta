import cv2
import threading
import time
import os
import datetime
import numpy as np
import logging
from flask import Flask, Response, render_template
import firebase_admin
from firebase_admin import credentials, db

# Local Imports
from app_detect import detect
import config

# =============================
# INITIALIZATION & DEBUG LOGGING
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

os.makedirs(config.SAVE_DIR, exist_ok=True)

# Define detection zones (Adjust these coordinates for your cameras)
ZONE_CAM1 = np.array([[100, 300], [500, 300], [600, 700], [50, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [400, 200], [400, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
            logger.info("Firebase Connected Successfully")
        except Exception as e:
            logger.error(f"Firebase Init Error: {e}")
            self.ref = None

    def report_violation(self, cam_name, count, frame):
        if self.ref:
            threading.Thread(target=self._upload_task, args=(cam_name, count, frame.copy()), daemon=True).start()

    def _upload_task(self, cam_name, count, frame):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{cam_name}_{timestamp}.jpg"
        filepath = os.path.join(config.SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        
        self.ref.push({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": cam_name,
            "vehicle_count": count,
            "status": "Illegal Parking Detected",
            "image_path": filename
        })

db_client = FirebaseHandler()

class VehicleTracker:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_seen = {} # {cam: {id: time}}
        self.violated_ids = {}
        self.last_capture = {} # {cam: {id: time}}
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}

    def is_in_zone(self, box, cam_name):
        zone = self.zones.get(cam_name)
        if zone is None: return False
        x1, y1, x2, y2 = box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return cv2.pointPolygonTest(zone, center, False) >= 0

    def process(self, cam_name, detections, frame):
        now = time.time()
        if cam_name not in self.first_seen:
            self.first_seen[cam_name], self.violated_ids[cam_name], self.last_capture[cam_name] = {}, set(), {}

        new_violation = False
        for obj_id, box in detections:
            if self.is_in_zone(box, cam_name):
                if obj_id not in self.first_seen[cam_name]:
                    self.first_seen[cam_name][obj_id] = now
                
                duration = now - self.first_seen[cam_name][obj_id]
                
                # Check if threshold exceeded
                if obj_id not in self.violated_ids[cam_name] and duration >= self.threshold:
                    self.violated_ids[cam_name].add(obj_id)
                    self.last_capture[cam_name][obj_id] = now
                    new_violation = True
                
                # Repeat capture logic
                elif obj_id in self.violated_ids[cam_name]:
                    if now - self.last_capture[cam_name].get(obj_id, 0) >= config.REPEAT_CAPTURE_INTERVAL:
                        self.last_capture[cam_name][obj_id] = now
                        new_violation = True
            else:
                self.first_seen[cam_name].pop(obj_id, None)

        if new_violation:
            db_client.report_violation(cam_name, len(detections), frame)

tracker = VehicleTracker(config.VIOLATION_TIME_THRESHOLD)

class StreamManager:
    def __init__(self, url, name):
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: time.sleep(0.01)

cam1 = StreamManager(config.CAM1_URL, "Camera_1")
cam2 = StreamManager(config.CAM2_URL, "Camera_2")
app = Flask(__name__)

def gen_frames():
    while True:
        active = []
        if cam1.frame is not None: active.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: active.append(("Camera_2", cam2.frame.copy()))
        
        if not active:
            time.sleep(0.1)
            continue

        # Hailo Inference
        raw_frames = [f for n, f in active]
        results = detect(raw_frames)

        processed_frames = []
        for i, res in enumerate(results):
            cam_name, frame = active[i]
            
            # Draw Zone for Debugging
            cv2.polylines(frame, [tracker.zones[cam_name]], True, (0, 255, 0), 2)
            
            # Format detections for tracker (using class index as temporary ID)
            formatted = []
            for box in res.boxes:
                formatted.append((box.cls[0], box.xyxy[0]))
                # Draw boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            tracker.process(cam_name, formatted, frame)
            processed_frames.append(cv2.resize(frame, (640, 480)))

        # Side-by-side display
        combined = cv2.hconcat(processed_frames) if len(processed_frames) > 1 else processed_frames[0]
        _, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
