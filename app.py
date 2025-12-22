import cv2
import threading
import time
import os
import numpy as np
import logging
from flask import Flask, Response, render_template, jsonify, request  # add jsonify, request
import json
from app_detect import detect
import config
import datetime

# --- Setup Logging & Folders ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# Settings management
SETTINGS_PATH = "settings.json"

# Default settings
VIOLATION_TIME_THRESHOLD = getattr(config, "VIOLATION_TIME_THRESHOLD", 10)
REPEAT_CAPTURE_INTERVAL = getattr(config, "REPEAT_CAPTURE_INTERVAL", 60)
current_settings = {
    "VIOLATION_TIME_THRESHOLD": VIOLATION_TIME_THRESHOLD,
    "REPEAT_CAPTURE_INTERVAL": REPEAT_CAPTURE_INTERVAL
}

def load_settings():
    global current_settings, VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r") as f:
            current_settings = json.load(f)
            VIOLATION_TIME_THRESHOLD = current_settings.get("VIOLATION_TIME_THRESHOLD", VIOLATION_TIME_THRESHOLD)
            REPEAT_CAPTURE_INTERVAL = current_settings.get("REPEAT_CAPTURE_INTERVAL", REPEAT_CAPTURE_INTERVAL)

def save_settings(data):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f)

load_settings()

class ByteTrackLite:
    def __init__(self):
        self.tracked_objects = {}
        self.frame_count = 0
        self.next_id = 0
        self.buffer = 30

    def get_iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, boxes, scores, clss):
        self.frame_count += 1
        new_tracks = {}
        for box, score, cid in zip(boxes, scores, clss):
            best_id, best_iou = None, 0.3
            for tid, t in self.tracked_objects.items():
                iou = self.get_iou(box, t['box'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                new_tracks[best_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                self.tracked_objects.pop(best_id, None)
            elif score >= config.DETECTION_THRESHOLD:
                new_tracks[self.next_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                self.next_id += 1

        for tid, t in self.tracked_objects.items():
            if self.frame_count - t['last_seen'] < self.buffer:
                new_tracks[tid] = t

        self.tracked_objects = new_tracks
        return {k: v for k, v in new_tracks.items() if v['last_seen'] == self.frame_count}

class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.timers = {}
        self.last_upload_time = {}
        # Define parking zones for each camera
        self.zones = {
            "Camera_1": np.array([[249, 242], [255, 404], [654, 426], [443, 261]]),
            "Camera_2": np.array([[46, 437], [453, 253], [664, 259], [678, 438]])
        }

    def process(self, name, res, frame):
        fh, fw = frame.shape[:2]
        cv2.polylines(frame, [self.zones[name]], True, (0, 255, 0), 2)
        
        pixel_boxes = [[b[0]*fw, b[1]*fh, b[2]*fw, b[3]*fh] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        now = time.time()

        for tid, d in tracked.items():
            x1, y1, x2, y2 = map(int, d['box'])
            label = CLASS_NAMES.get(d['cls'], "OBJ")
            center = ((x1+x2)//2, (y1+y2)//2)

            # Person detection (Yellow box, no timer)
            if d['cls'] == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                continue

            # Vehicle in Zone logic
            if cv2.pointPolygonTest(self.zones[name], center, False) >= 0:
                self.timers.setdefault((name, tid), now)
                dur = int(now - self.timers[(name, tid)])
                
                is_violation = dur >= config.VIOLATION_TIME_THRESHOLD
                color = (0, 0, 255) if is_violation else (0, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{tid}: {dur}s", (x1, y1-8), 0, 0.6, color, 2)

                if is_violation:
                    last_up = self.last_upload_time.get((name, tid), 0)
                    if now - last_up > config.REPEAT_CAPTURE_INTERVAL:
                        self.log_violation(name, tid, label, frame)
                        self.last_upload_time[(name, tid)] = now
            else:
                self.timers.pop((name, tid), None)

    def log_violation(self, cam, tid, label, frame):
        ts = int(time.time())
        # Get current date and weekday
        now = datetime.datetime.now()
        date_folder = now.strftime("<%B %d, %Y (%A)>")
        # Create the date folder if it doesn't exist
        date_dir = os.path.join(config.SAVE_DIR, date_folder)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        filename = f"{cam}_{tid}_{ts}.jpg"
        path = os.path.join(date_dir, filename)
        cv2.imwrite(path, frame)
        logger.info(f"Violation Logged: {label} on {cam} (saved to {path})")

class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.last_update = None  # Track last frame update time
        self.reconnect_event = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            if self.reconnect_event.is_set():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                self.reconnect_event.clear()
            ret, f = self.cap.read()
            if ret:
                self.frame = f
                self.last_update = time.time()
            else:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

    def is_online(self, timeout=2.0):
        """Returns True if the stream has updated recently."""
        return self.last_update is not None and (time.time() - self.last_update) < timeout

    def reconnect(self):
        self.reconnect_event.set()

app = Flask(__name__)
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

def gen():
    offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
    while True:
        active = []
        online_cameras = set()
        if c1.frame is not None and c1.is_online():
            active.append(("Camera_1", c1.frame.copy()))
            online_cameras.add("Camera_1")
        if c2.frame is not None and c2.is_online():
            active.append(("Camera_2", c2.frame.copy()))
            online_cameras.add("Camera_2")
        
        # Clear timers for offline cameras to avoid stale violation timing/capture
        for cam_name in ["Camera_1", "Camera_2"]:
            if cam_name not in online_cameras:
                # Remove all timers and last_upload_time for this camera
                monitor.timers = {k: v for k, v in monitor.timers.items() if k[0] != cam_name}
                monitor.last_upload_time = {k: v for k, v in monitor.last_upload_time.items() if k[0] != cam_name}

        # If both are offline, show offline placeholder
        if not active:
            combined = np.hstack([offline_placeholder, offline_placeholder])
            _, buf = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        try:
            results = detect([f for _, f in active])
            out = []
            for i, res in enumerate(results):
                name, frame = active[i]
                # Only process if camera is online (extra safety)
                if name in online_cameras:
                    monitor.process(name, res, frame)
                out.append(cv2.resize(frame, (640, 480)))

            if len(out) == 1:
                out.append(offline_placeholder)
            combined = cv2.hconcat(out)
            _, buf = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Gen Error: {e}")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/violations.html')
def violations_page():
    return render_template('violations.html')

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

@app.route('/api/reconnect/<camera>', methods=['POST'])
def reconnect_camera(camera):
    if camera == "Camera_1":
        c1.reconnect()
        return jsonify({"success": True, "message": "Camera_1 reconnect triggered"})
    elif camera == "Camera_2":
        c2.reconnect()
        return jsonify({"success": True, "message": "Camera_2 reconnect triggered"})
    else:
        return jsonify({"success": False, "message": "Unknown camera"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
