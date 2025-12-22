import cv2
import threading
import time
import os
import numpy as np
import logging
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
import json
from app_detect import detect
import config
import datetime
import queue

# --- Setup Logging & Folders ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# Settings management (now only via config.py)
def get_current_settings():
    return {
        "VIOLATION_TIME_THRESHOLD": getattr(config, "VIOLATION_TIME_THRESHOLD", 10),
        "REPEAT_CAPTURE_INTERVAL": getattr(config, "REPEAT_CAPTURE_INTERVAL", 60)
    }

def update_config_py(new_settings):
    import re
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        lines = f.readlines()
    def replace_line(key, value):
        pattern = re.compile(rf"^{key}\s*=\s*.*$")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key} = {value}\n"
                return
        # If not found, append
        lines.append(f"{key} = {value}\n")
    replace_line("VIOLATION_TIME_THRESHOLD", new_settings["VIOLATION_TIME_THRESHOLD"])
    replace_line("REPEAT_CAPTURE_INTERVAL", new_settings["REPEAT_CAPTURE_INTERVAL"])
    with open(config_path, "w") as f:
        f.writelines(lines)
    # Reload config module
    import importlib
    importlib.reload(config)

# --- Add: Zone configuration persistence ---
ZONE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "zone_config.json")

def load_zones():
    if os.path.exists(ZONE_CONFIG_PATH):
        with open(ZONE_CONFIG_PATH, "r") as f:
            return json.load(f)
    # Default zones (as in original code)
    return {
        "Camera_1": [[249, 242], [255, 404], [654, 426], [443, 261]],
        "Camera_2": [[46, 437], [453, 253], [664, 259], [678, 438]]
    }

def save_zones(zones):
    with open(ZONE_CONFIG_PATH, "w") as f:
        json.dump(zones, f)

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
        # Load zones from config
        loaded_zones = load_zones()
        self.zones = {
            "Camera_1": np.array(loaded_zones.get("Camera_1"), np.int32),
            "Camera_2": np.array(loaded_zones.get("Camera_2"), np.int32)
        }

    def update_zones(self, new_zones):
        self.zones = {
            "Camera_1": np.array(new_zones.get("Camera_1"), np.int32),
            "Camera_2": np.array(new_zones.get("Camera_2"), np.int32)
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
        now = datetime.datetime.now()
        date_folder = now.strftime("%B %d, %Y (%A)")
        date_dir = os.path.join(config.SAVE_DIR, date_folder)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        # Format time as HH_MM_ss for filename safety
        time_str = now.strftime("%H_%M_%S")
        filename = f"{cam}-{time_str}.jpg"
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

    @property
    def reconnecting(self):
        # True if not online, False if online
        return not self.is_online()

    def reconnect(self):
        self.reconnect_event.set()

app = Flask(__name__)
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

# Shared latest processed frames for each camera
latest_frames = {"Camera_1": None, "Camera_2": None}
latest_frames_lock = threading.Lock()

def background_detection_loop():
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
                monitor.timers = {k: v for k, v in monitor.timers.items() if k[0] != cam_name}
                monitor.last_upload_time = {k: v for k, v in monitor.last_upload_time.items() if k[0] != cam_name}

        try:
            if active:
                results = detect([f for _, f in active])
                with latest_frames_lock:
                    for i, res in enumerate(results):
                        name, frame = active[i]
                        monitor.process(name, res, frame)
                        # Store the processed frame (resize for consistency)
                        latest_frames[name] = cv2.resize(frame, (640, 480))
            else:
                with latest_frames_lock:
                    latest_frames["Camera_1"] = offline_placeholder
                    latest_frames["Camera_2"] = offline_placeholder
        except Exception as e:
            logger.error(f"Background Detection Error: {e}")
        time.sleep(0.03)

# Start background detection thread
threading.Thread(target=background_detection_loop, daemon=True).start()

def gen():
    while True:
        with latest_frames_lock:
            frame1 = latest_frames.get("Camera_1")
            frame2 = latest_frames.get("Camera_2")
            out = []
            if frame1 is not None:
                out.append(frame1)
            if frame2 is not None:
                out.append(frame2)
            if not out:
                # fallback placeholder
                offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
                out = [offline_placeholder, offline_placeholder]
            elif len(out) == 1:
                offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
                out.append(offline_placeholder)
            combined = cv2.hconcat(out)
            _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

def gen_single(cam, cam_name):
    offline_placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(offline_placeholder, f"{cam_name} OFFLINE", (120, 360), 0, 2.2, (0,0,255), 5, cv2.LINE_AA)
    while True:
        with latest_frames_lock:
            frame = latest_frames.get(cam_name)
            if frame is not None:
                out = cv2.resize(frame, (1280, 720))
            else:
                out = offline_placeholder
            _, buf = cv2.imencode('.jpg', out)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    # (Optional: keep for backward compatibility, or remove if not needed)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c1')
def video_feed_c1():
    return Response(gen_single(c1, "Camera_1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c2')
def video_feed_c2():
    return Response(gen_single(c2, "Camera_2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/violations.html')
def violations_page():
    return render_template('violations.html')

@app.route('/zone_selector.html')
def zone_selector_page():
    return render_template('zone_selector.html')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(get_current_settings())

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    update_config_py(data)
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

@app.route('/api/camera_status')
def camera_status():
    return jsonify({
        "Camera_1": {
            "reconnecting": c1.reconnecting
        },
        "Camera_2": {
            "reconnecting": c2.reconnecting
        }
    })

@app.route('/api/zones', methods=['GET'])
def get_zones():
    return jsonify(load_zones())

@app.route('/api/zones', methods=['POST'])
def set_zones():
    data = request.get_json()
    save_zones(data)
    monitor.update_zones(data)
    return jsonify({"success": True})

@app.route('/snapshot/<camera>')
def snapshot(camera):
    with latest_frames_lock:
        frame = latest_frames.get(camera)
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            return Response(buf.tobytes(), mimetype='image/jpeg')
        else:
            offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
            _, buf = cv2.imencode('.jpg', offline_placeholder)
            return Response(buf.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)