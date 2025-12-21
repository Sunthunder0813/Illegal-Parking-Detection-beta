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

# Local Imports
from app_detect import detect
import config

# =============================
# INITIALIZATION & LOGGING
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

os.makedirs(config.SAVE_DIR, exist_ok=True)

# Polygons for violation detection
ZONE_CAM1 = np.array([[100, 300], [500, 300], [600, 700], [50, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [400, 200], [400, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
            logger.info("Firebase Connected")
        except Exception as e:
            logger.error(f"Firebase Init Error: {e}")
            self.ref = None

    def report_violation(self, cam_name, count, frame):
        if self.ref:
            threading.Thread(target=self._upload_task, args=(cam_name, count, frame.copy()), daemon=True).start()

    def _upload_task(self, cam_name, count, frame):
        try:
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
        except Exception as e:
            logger.error(f"Firebase Upload Failed: {e}")

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
                
                # Visual Indicator for Timer
                x1, y1, x2, y2 = map(int, box)
                timer_text = f"Parked: {int(duration)}s"
                color = (0, 0, 255) if duration >= self.threshold else (0, 255, 255)
                cv2.putText(frame, timer_text, (x1, y1 - 30), 0, 0.5, color, 2)

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
        self.url = url
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.active = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                self.frame = None # Explicitly set to None to trigger placeholder logic
                logger.warning(f"Lost connection to {self.name}. Reconnecting...")
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.url)

cam1 = StreamManager(config.CAM1_URL, "Camera_1")
cam2 = StreamManager(config.CAM2_URL, "Camera_2")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <body style="background: #000; color: white; text-align: center; font-family: sans-serif;">
                <h1>Hailo-8L Dual Camera Parking Monitor</h1>
                <div style="margin: 20px;">
                    <img src="/video_feed" style="width: 90%; border: 5px solid #333; border-radius: 10px;">
                </div>
                <p>Status: Monitoring live RTSP streams</p>
            </body>
        </html>
    """)

def gen_frames():
    while True:
        available_streams = []
        if cam1.frame is not None: available_streams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: available_streams.append(("Camera_2", cam2.frame.copy()))
        
        if not available_streams:
            # Both cameras offline: Show warning
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black, "NO CAMERA CONNECTION", (120, 240), 0, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', black)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue

        # AI Detection
        raw_frames = [f for n, f in available_streams]
        results = detect(raw_frames)

        processed_map = {}
        for i, res in enumerate(results):
            cam_name, frame = available_streams[i]
            cv2.polylines(frame, [tracker.zones[cam_name]], True, (0, 255, 0), 2)
            
            formatted_detections = []
            if res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy()
                for box, obj_id in zip(boxes, ids):
                    formatted_detections.append((int(obj_id), box))
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID:{int(obj_id)}", (x1, y1-10), 0, 0.5, (255,0,0), 2)

            tracker.process(cam_name, formatted_detections, frame)
            processed_map[cam_name] = cv2.resize(frame, (640, 480))

        # Layout Logic: Maintain dual view even if one camera is gone
        final_layout = []
        for name in ["Camera_1", "Camera_2"]:
            if name in processed_map:
                final_layout.append(processed_map[name])
            else:
                # Placeholder for offline camera
                offline_box = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(offline_box, f"{name} OFFLINE", (180, 240), 0, 0.8, (0, 0, 255), 2)
                final_layout.append(offline_box)

        combined = cv2.hconcat(final_layout)
        _, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
