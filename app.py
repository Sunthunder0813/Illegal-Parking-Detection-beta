import cv2
import threading
import time
import os
import numpy as np
import logging
from flask import Flask, Response, render_template, jsonify, request
import json
from app_detect import detect
import config
import datetime
import importlib

# --- Setup Logging & Folders ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# --- Settings management ---
def get_current_settings():
    return {
        "VIOLATION_TIME_THRESHOLD": getattr(config, "VIOLATION_TIME_THRESHOLD", 10),
        "REPEAT_CAPTURE_INTERVAL": getattr(config, "REPEAT_CAPTURE_INTERVAL", 60),
        "PARKING_ZONES": getattr(config, "PARKING_ZONES", {})
    }

def update_config_py(new_settings):
    import re
    import json as pyjson
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        lines = f.readlines()
    def replace_line(key, value):
        pattern = re.compile(rf"^{key}\s*=\s*.*$")
        for i, line in enumerate(lines):
            if pattern.match(line):
                if key == "PARKING_ZONES":
                    lines[i] = f"{key} = {pyjson.dumps(value)}\n"
                else:
                    lines[i] = f"{key} = {value}\n"
                return
        lines.append(f"{key} = {pyjson.dumps(value) if key == 'PARKING_ZONES' else value}\n")
    
    if "VIOLATION_TIME_THRESHOLD" in new_settings:
        replace_line("VIOLATION_TIME_THRESHOLD", new_settings["VIOLATION_TIME_THRESHOLD"])
    if "REPEAT_CAPTURE_INTERVAL" in new_settings:
        replace_line("REPEAT_CAPTURE_INTERVAL", new_settings["REPEAT_CAPTURE_INTERVAL"])
    if "PARKING_ZONES" in new_settings:
        current_zones = getattr(config, "PARKING_ZONES", {})
        updated_zones = current_zones.copy()
        for cam, val in new_settings["PARKING_ZONES"].items():
            if val is None:
                updated_zones.pop(cam, None)
            else:
                updated_zones[cam] = val
        replace_line("PARKING_ZONES", updated_zones)
        
    with open(config_path, "w") as f:
        f.writelines(lines)
    importlib.reload(config)

# --- Tracking Logic ---
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
        self.reload_zones()

    def reload_zones(self):
        importlib.reload(config)
        self.zones = {
            cam: np.array(points)
            for cam, points in getattr(config, "PARKING_ZONES", {}).items()
        }

    def process(self, name, res, frame):
        self.reload_zones()
        if name not in self.zones: return
        
        fh, fw = frame.shape[:2]
        cv2.polylines(frame, [self.zones[name]], True, (0, 0, 255), 2)
        
        pixel_boxes = [[b[0]*fw, b[1]*fh, b[2]*fw, b[3]*fh] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        now = time.time()

        for tid, d in tracked.items():
            x1, y1, x2, y2 = map(int, d['box'])
            label = CLASS_NAMES.get(d['cls'], "OBJ")
            center = ((x1+x2)//2, (y1+y2)//2)

            if d['cls'] == 0: # Person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                continue

            in_zone = cv2.pointPolygonTest(self.zones[name], center, False) >= 0
            if in_zone:
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                self.timers.pop((name, tid), None)

    def log_violation(self, cam, tid, label, frame):
        now = datetime.datetime.now()
        date_folder = now.strftime("%B %d, %Y (%A)")
        date_dir = os.path.join(config.SAVE_DIR, date_folder)
        if not os.path.exists(date_dir): os.makedirs(date_dir)
        path = os.path.join(date_dir, f"{cam}-{now.strftime('%H_%M_%S')}.jpg")
        cv2.imwrite(path, frame)
        logger.info(f"Violation Logged: {label} on {cam}")

# --- Thread-Safe Stream Class ---
class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.last_update = 0
        self.reconnecting = False
        self.read_lock = threading.Lock()
        self.running = True
        self.reconnect_event = threading.Event()
        # Single thread for all Capture IO
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while self.running:
            if self.reconnect_event.is_set():
                self._do_reconnect()

            ret, f = self.cap.read()
            if ret:
                with self.read_lock:
                    self.frame = f
                    self.last_update = time.time()
                self.reconnecting = False
            else:
                self.reconnecting = True
                time.sleep(1)
                self._do_reconnect()
            time.sleep(0.01)

    def _do_reconnect(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.url)
        self.reconnect_event.clear()

    def is_online(self):
        return (time.time() - self.last_update) < 3.0

    def get_frame(self):
        with self.read_lock:
            return self.frame.copy() if self.frame is not None else None

    def reconnect(self):
        self.reconnect_event.set()

# --- Global States ---
app = Flask(__name__)
monitor = ParkingMonitor()
c1 = Stream(config.CAM1_URL)
c2 = Stream(config.CAM2_URL)

latest_processed = {"Camera_1": None, "Camera_2": None}
proc_lock = threading.Lock()

def processing_worker(cam_name, stream):
    while True:
        frame = stream.get_frame()
        if frame is not None:
            # Run detection
            results = detect([frame])
            if results:
                # Process zones and labels
                monitor.process(cam_name, results[0], frame)
                with proc_lock:
                    latest_processed[cam_name] = frame
        time.sleep(0.1) # Frequency of AI analysis

# Start Background Threads
threading.Thread(target=processing_worker, args=("Camera_1", c1), daemon=True).start()
threading.Thread(target=processing_worker, args=("Camera_2", c2), daemon=True).start()

# --- Flask Routes ---
def gen_single(stream, cam_name):
    while True:
        with proc_lock:
            frame = latest_processed.get(cam_name)
        
        # If processing is slow, fallback to raw stream
        if frame is None:
            frame = stream.get_frame()
            
        if frame is not None:
            frame = cv2.resize(frame, (1280, 720))
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"{cam_name} OFFLINE", (400, 360), 0, 1.5, (0,0,255), 3)

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index(): return render_template("index.html")

@app.route('/video_feed_c1')
def video_feed_c1(): return Response(gen_single(c1, "Camera_1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c2')
def video_feed_c2(): return Response(gen_single(c2, "Camera_2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        update_config_py(request.get_json())
        return jsonify({"success": True})
    return jsonify(get_current_settings())

@app.route('/api/camera_status')
def camera_status():
    return jsonify({
        "Camera_1": {"online": c1.is_online(), "reconnecting": c1.reconnecting},
        "Camera_2": {"online": c2.is_online(), "reconnecting": c2.reconnecting}
    })

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
