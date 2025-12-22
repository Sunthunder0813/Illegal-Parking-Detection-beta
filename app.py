import cv2
import threading
import time
import os
import numpy as np
import logging
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response, render_template_string
from app_detect import detect
import config

# --- Setup Logging & Folders ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

# --- Firebase ---
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
    ref = db.reference('violations')
except Exception as e:
    logger.error(f"Firebase Init Error: {e}")
    ref = None

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

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
        filename = f"{cam}_{tid}_{ts}.jpg"
        path = os.path.join(config.SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        if ref:
            try:
                ref.push({
                    'camera': cam, 
                    'object_id': tid, 
                    'type': label, 
                    'timestamp': ts, 
                    'local_path': path
                })
                logger.info(f"Violation Logged: {label} on {cam}")
            except Exception as e:
                logger.error(f"Firebase Push Error: {e}")

class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret:
                self.frame = f
            else:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

app = Flask(__name__)
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

def gen():
    while True:
        active = []
        if c1.frame is not None: active.append(("Camera_1", c1.frame.copy()))
        if c2.frame is not None: active.append(("Camera_2", c2.frame.copy()))
        
        if not active:
            time.sleep(0.01)
            continue

        try:
            results = detect([f for _, f in active])
            out = []
            for i, res in enumerate(results):
                name, frame = active[i]
                monitor.process(name, res, frame)
                out.append(cv2.resize(frame, (640, 480)))

            combined = cv2.hconcat(out) if len(out) == 2 else out[0]
            _, buf = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Gen Error: {e}")

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <head><title>Parking Monitor</title></head>
            <body style="background:#111; color:white; text-align:center;">
                <h1>Illegal Parking Detection - Live</h1>
                <img src="/video_feed" style="border:2px solid #444; width:90%;">
                <p>Status: Monitoring Camera 1 and Camera 2</p>
            </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
