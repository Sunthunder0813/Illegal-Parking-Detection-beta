import cv2
import threading
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, render_template
import datetime
import sys
import numpy as np

# Assuming these are imported from your local files
from detect import NAMES, detect 
from config import (CAM1_URL, CAM2_URL, SAVE_DIR, DATABASE_URL, 
                    FIREBASE_KEY_PATH, UPLOAD_COOLDOWN, VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL)

# =============================
# CONFIG & FIREBASE
# =============================
os.makedirs(SAVE_DIR, exist_ok=True)

# DEFINE YOUR ILLEGAL ZONES (Normalized coordinates 0.0 to 1.0 or pixel coordinates)
# Format: np.array([[x1, y1], [x2, y2], ...], np.int32)
# For this example, I'm using placeholder coordinates. Update these to match your camera view.
ZONE_CAM1 = np.array([[87, 442], [44, 213], [467, 241], [654, 446]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [400, 200], [400, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
            self.ref = db.reference('violations_history')
            print("Connected to Firebase Realtime Database")
        except Exception as e:
            print(f"Firebase Error: {e}")

    def report_violation(self, cam_names, counts, merged_frame):
        total_count = sum(counts)
        if total_count > 0:
            threading.Thread(target=self._process_local_save, 
                             args=(cam_names, counts, merged_frame.copy()), 
                             daemon=True).start()

    def _process_local_save(self, cam_names, counts, merged_frame):
        try:
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
            print(f"Logged Violation: {cam_names}")
        except Exception as e:
            print(f"Local Save Error: {e}")

db_client = FirebaseHandler()

# =============================
# VEHICLE TRACKER WITH ROI LOGIC
# =============================
class VehicleTracker:
    def __init__(self, threshold):
        self.threshold = threshold  # Seconds
        self.first_seen = {}        # {cam_name: {vid: timestamp}}
        self.violated_ids = {}      # {cam_name: set(vid)}
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}
        # Track last capture time for each violator
        self.last_capture_time = {} # {cam_name: {vid: last_capture_time}}

    def is_box_in_zone(self, box_xyxy, cam_name):
        zone = self.zones.get(cam_name)
        if zone is None:
            return True
        x1, y1, x2, y2 = box_xyxy
        # Check corners and center
        points = [
            (int(x1), int(y1)),  # top-left
            (int(x2), int(y1)),  # top-right
            (int(x1), int(y2)),  # bottom-left
            (int(x2), int(y2)),  # bottom-right
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # center
        ]
        return any(cv2.pointPolygonTest(zone, pt, False) >= 0 for pt in points)

    def update(self, cam_name, current_detections):
        """
        current_detections: list of (vid, box_xyxy)
        Returns:
            new_violations_count: int
            repeat_violators: list of vid
        """
        now = time.time()
        if cam_name not in self.first_seen:
            self.first_seen[cam_name] = {}
            self.violated_ids[cam_name] = set()
            self.last_capture_time[cam_name] = {}

        active_vids = [d[0] for d in current_detections]
        # Cleanup old tracking data
        for vid in list(self.first_seen[cam_name].keys()):
            if vid not in active_vids:
                del self.first_seen[cam_name][vid]
                self.violated_ids[cam_name].discard(vid)
                self.last_capture_time[cam_name].pop(vid, None)

        new_violations_count = 0
        repeat_violators = []
        for vid, box_xyxy in current_detections:
            if self.is_box_in_zone(box_xyxy, cam_name):
                if vid not in self.first_seen[cam_name]:
                    self.first_seen[cam_name][vid] = now
                duration = now - self.first_seen[cam_name][vid]
                if vid not in self.violated_ids[cam_name] and duration >= self.threshold:
                    print(f"[DEBUG] Violation: {cam_name} vehicle {vid} in zone for {duration:.2f}s")
                    self.violated_ids[cam_name].add(vid)
                    new_violations_count += 1
                    self.last_capture_time[cam_name][vid] = now
                elif vid in self.violated_ids[cam_name]:
                    # Already violated, check if enough time passed for repeat capture
                    last_time = self.last_capture_time[cam_name].get(vid, 0)
                    if now - last_time >= REPEAT_CAPTURE_INTERVAL:
                        repeat_violators.append(vid)
                        self.last_capture_time[cam_name][vid] = now
            else:
                # If vehicle leaves the zone, reset timer
                self.first_seen[cam_name].pop(vid, None)
                self.violated_ids[cam_name].discard(vid)
                self.last_capture_time[cam_name].pop(vid, None)
        return new_violations_count, repeat_violators

vehicle_tracker = VehicleTracker(VIOLATION_TIME_THRESHOLD)

# =============================
# STREAM HANDLING
# =============================
class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: time.sleep(0.1)

    def read(self): return self.frame

cam1 = RTSPStream(CAM1_URL)
cam2 = RTSPStream(CAM2_URL)

# =============================
# FLASK & PROCESSING
# =============================
app = Flask(__name__)
violation_captured_flag = False

def gen_frames():
    global violation_captured_flag
    last_upload_time = 0
    ILLEGAL_CLASSES = {"car", "truck", "bus", "motorcycle"}

    while True:
        cam_frames = [("Camera_1", cam1.read()), ("Camera_2", cam2.read())]
        active_frames = [(n, f) for n, f in cam_frames if f is not None]
        
        if not active_frames:
            time.sleep(0.01)
            continue

        frame_list = [f for n, f in active_frames]
        results = detect(frame_list)
        
        counts = []
        processed_plots = []
        repeat_violators_per_cam = []

        for i, res in enumerate(results):
            cam_name = active_frames[i][0]
            current_vids_with_box = []
            
            # Draw Zone on frame for visualization (filled red with transparency)
            zone_pts = vehicle_tracker.zones.get(cam_name)
            overlay = frame_list[i].copy()
            cv2.fillPoly(overlay, [zone_pts], color=(0, 0, 255))
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame_list[i], 1 - alpha, 0, frame_list[i])
            cv2.polylines(frame_list[i], [zone_pts], True, (0, 0, 255), 2)

            for box in res.boxes:
                cls_idx = int(box.cls[0])
                label = NAMES[cls_idx]
                if label in ILLEGAL_CLASSES:
                    vid = int(box.id[0]) if box.id is not None else hash(tuple(box.xyxy[0].tolist()))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_xyxy = (x1, y1, x2, y2)
                    current_vids_with_box.append((vid, box_xyxy))

            new_violations, repeat_violators = vehicle_tracker.update(cam_name, current_vids_with_box)
            counts.append(new_violations)
            repeat_violators_per_cam.append(repeat_violators)

            # Render labels on frame
            annotated_frame = res.plot()
            processed_plots.append(annotated_frame)

        # Check for new violations (with cooldown) or repeat violators (no cooldown)
        trigger_new = any(c > 0 for c in counts) and (time.time() - last_upload_time > UPLOAD_COOLDOWN)
        trigger_repeat = any(len(r) > 0 for r in repeat_violators_per_cam)
        if trigger_new or trigger_repeat:
            merged_img = cv2.hconcat(processed_plots)
            # For repeat captures, set counts to number of repeat violators per camera
            if trigger_repeat and not trigger_new:
                repeat_counts = [len(r) for r in repeat_violators_per_cam]
                db_client.report_violation([n for n, f in active_frames], repeat_counts, merged_img)
            else:
                db_client.report_violation([n for n, f in active_frames], counts, merged_img)
            last_upload_time = time.time()
            violation_captured_flag = True

        # Streaming
        final_view = cv2.hconcat(processed_plots)
        _, buffer = cv2.imencode('.jpg', final_view)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    global violation_captured_flag
    msg = "Violation captured!" if violation_captured_flag else None
    violation_captured_flag = False
    return render_template('index.html', notif=msg)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)