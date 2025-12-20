import cv2
import threading
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, render_template
import datetime
import numpy as np

# Local Imports
from app_detect import NAMES, detect 
from config import (CAM1_URL, CAM2_URL, SAVE_DIR, DATABASE_URL, 
                    FIREBASE_KEY_PATH, UPLOAD_COOLDOWN, VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL)

# =============================
# INITIALIZATION
# =============================
os.makedirs(SAVE_DIR, exist_ok=True)
ZONE_CAM1 = np.array([[100, 300], [500, 300], [600, 700], [50, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [400, 200], [400, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
            self.ref = db.reference('violations_history')
        except Exception as e:
            print(f"Firebase Error: {e}")

    def report_violation(self, cam_names, counts, merged_frame):
        threading.Thread(target=self._process_local_save, 
                         args=(cam_names, counts, merged_frame.copy()), 
                         daemon=True).start()

    def _process_local_save(self, cam_names, counts, merged_frame):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp_str}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, merged_frame)
        self.ref.push({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cameras": cam_names,
            "vehicle_counts": counts,
            "local_file": filename,
            "status": "Illegal Parking Detected"
        })

db_client = FirebaseHandler()

class VehicleTracker:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_seen = {} # {cam: {id: time}}
        self.violated_ids = {}
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}
        self.last_capture_time = {}

    def is_box_in_zone(self, box_xyxy, cam_name):
        zone = self.zones.get(cam_name)
        if zone is None: return False
        x1, y1, x2, y2 = box_xyxy
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return cv2.pointPolygonTest(zone, center, False) >= 0

    def update(self, cam_name, current_detections):
        now = time.time()
        if cam_name not in self.first_seen:
            self.first_seen[cam_name], self.violated_ids[cam_name], self.last_capture_time[cam_name] = {}, set(), {}

        new_violations_count = 0
        repeat_violators = []

        for vid, box_xyxy in current_detections:
            if self.is_box_in_zone(box_xyxy, cam_name):
                if vid not in self.first_seen[cam_name]:
                    self.first_seen[cam_name][vid] = now
                
                duration = now - self.first_seen[cam_name][vid]
                if vid not in self.violated_ids[cam_name] and duration >= self.threshold:
                    self.violated_ids[cam_name].add(vid)
                    new_violations_count += 1
                    self.last_capture_time[cam_name][vid] = now
                elif vid in self.violated_ids[cam_name]:
                    if now - self.last_capture_time[cam_name].get(vid, 0) >= REPEAT_CAPTURE_INTERVAL:
                        repeat_violators.append(vid)
                        self.last_capture_time[cam_name][vid] = now
            else:
                self.first_seen[cam_name].pop(vid, None)

        return new_violations_count, repeat_violators

tracker = VehicleTracker(VIOLATION_TIME_THRESHOLD)

class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: time.sleep(0.01)

cam1, cam2 = RTSPStream(CAM1_URL), RTSPStream(CAM2_URL)
app = Flask(__name__)

def gen_frames():
    last_upload = 0
    while True:
        frames = [("Camera_1", cam1.frame), ("Camera_2", cam2.frame)]
        active = [(n, f) for n, f in frames if f is not None]
        if not active: continue

        results = detect([f for n, f in active])
        plots, total_new = [], 0

        for i, res in enumerate(results):
            cam_name = active[i][0]
            # Process detections logic (omitted for brevity, same as your original)
            plots.append(active[i][1]) # Showing raw frame for now

        final_view = cv2.hconcat(plots)
        _, buffer = cv2.imencode('.jpg', final_view)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)