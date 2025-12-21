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
os.makedirs(config.SAVE_DIR, exist_ok=True)

# Zones (Ensure these are within your camera's resolution, e.g., 1280x720)
ZONE_CAM1 = np.array([[100, 300], [1100, 300], [1100, 700], [100, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [1000, 200], [1000, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
            logger.info("Firebase Connected Successfully")
        except Exception as e:
            logger.error(f"Firebase Init Error: {e}"); self.ref = None

    def report_violation(self, cam_name, count, frame):
        if self.ref:
            threading.Thread(target=self._upload_task, args=(cam_name, count, frame.copy()), daemon=True).start()

    def _upload_task(self, cam_name, count, frame):
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{cam_name}_{ts}.jpg"
            fpath = os.path.join(config.SAVE_DIR, fname)
            cv2.imwrite(fpath, frame)
            self.ref.push({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_name,
                "vehicle_count": count,
                "status": "Illegal Parking Detected",
                "image_path": fname
            })
            logger.info(f"Violation Uploaded: {cam_name}")
        except Exception as e: logger.error(f"Firebase Error: {e}")

db_client = FirebaseHandler()

class VehicleTracker:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_seen = {} 
        self.violated_ids = {}
        self.last_capture = {} 
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
                x1, y1, x2, y2 = map(int, box)
                
                # Highlight active vehicle
                color = (0, 0, 255) if duration >= self.threshold else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"VEHICLE {obj_id}: {int(duration)}s", (x1, y1 - 10), 0, 0.7, color, 2)

                if obj_id not in self.violated_ids[cam_name] and duration >= self.threshold:
                    self.violated_ids[cam_name].add(obj_id)
                    self.last_capture[cam_name][obj_id] = now
                    new_violation = True
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
        self.url, self.name, self.frame, self.active = url, name, None, True
        self.cap = cv2.VideoCapture(url)
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.active:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else:
                self.frame = None
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.url)

cam1 = StreamManager(config.CAM1_URL, "Camera_1")
cam2 = StreamManager(config.CAM2_URL, "Camera_2")
app = Flask(__name__)

def gen_frames():
    while True:
        streams = []
        if cam1.frame is not None: streams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: streams.append(("Camera_2", cam2.frame.copy()))
        
        if not streams:
            time.sleep(0.1); continue

        results = detect([f for n, f in streams])

        processed = {}
        for i, res in enumerate(results):
            cam_name, frame = streams[i]
            h, w = frame.shape[:2]
            # Draw Monitoring Zone
            cv2.polylines(frame, [tracker.zones[cam_name]], True, (0, 255, 0), 3)
            
            formatted = []
            if res.boxes is not None and res.boxes.id is not None:
                for box, obj_id in zip(res.boxes.xyxy, res.boxes.id):
                    x1, y1, x2, y2 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
                    formatted.append((int(obj_id), [x1, y1, x2, y2]))

            tracker.process(cam_name, formatted, frame)
            processed[cam_name] = cv2.resize(frame, (640, 480))

        layout = [processed.get(n, np.zeros((480, 640, 3), np.uint8)) for n in ["Camera_1", "Camera_2"]]
        combined = cv2.hconcat(layout)
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("<h1>Live Detection</h1><img src='/video_feed' width='100%'>")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
