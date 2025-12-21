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

# Initialize Firebase
try:
    cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
    fb_db = db.reference('violations')
except Exception as e:
    print(f"Firebase Init Error: {e}")
    fb_db = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

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
        current_frame_tracks = {}
        
        for box, score, cid in zip(boxes, scores, clss):
            best_id, best_iou = None, 0.3
            for tid, t in self.tracked_objects.items():
                iou = self.get_iou(box, t['box'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                current_frame_tracks[best_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                del self.tracked_objects[best_id]
            elif score >= 0.3:
                current_frame_tracks[self.next_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                self.next_id += 1

        # Keep tracks that were not seen this frame but are within buffer
        for tid, t in list(self.tracked_objects.items()):
            if self.frame_count - t['last_seen'] < self.buffer:
                current_frame_tracks[tid] = t
        
        self.tracked_objects = current_frame_tracks
        return {k: v for k, v in current_frame_tracks.items() if v['last_seen'] == self.frame_count}

class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.timers = {}
        self.last_upload = {}
        self.zones = {
            "Camera_1": np.array([[249, 242], [255, 404], [654, 426], [443, 261]]),
            "Camera_2": np.array([[200, 200], [1000, 200], [1000, 600], [200, 600]])
        }

    def handle_violation(self, name, tid, label, frame):
        key = (name, tid)
        now = time.time()
        if now - self.last_upload.get(key, 0) > config.REPEAT_CAPTURE_INTERVAL:
            timestamp = int(now)
            filename = f"violation_{name}_{tid}_{timestamp}.jpg"
            filepath = os.path.join(config.SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            
            if fb_db:
                fb_db.push({
                    'camera': name,
                    'object_id': tid,
                    'type': label,
                    'timestamp': timestamp,
                    'image_path': filepath
                })
            self.last_upload[key] = now
            logger.info(f"Violation Logged: {name} ID:{tid}")

    def process(self, name, res, frame):
        fh, fw = frame.shape[:2]
        cv2.polylines(frame, [self.zones[name]], True, (0, 255, 0), 2)
        
        pixel_boxes = [[b[0]*fw, b[1]*fh, b[2]*fw, b[3]*fh] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        now = time.time()

        for tid, d in tracked.items():
            x1, y1, x2, y2 = map(int, d['box'])
            cid = d['cls']
            label = CLASS_NAMES.get(cid, "OBJ")
            center = ((x1+x2)//2, (y1+y2)//2)

            if cid == 0: # Person - Just Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                continue

            if cv2.pointPolygonTest(self.zones[name], center, False) >= 0:
                self.timers.setdefault((name, tid), now)
                dur = int(now - self.timers[(name, tid)])
                
                is_violation = dur >= config.VIOLATION_TIME_THRESHOLD
                color = (0, 0, 255) if is_violation else (0, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{tid}: {dur}s", (x1, y1-8), 0, 0.6, color, 2)
                
                if is_violation:
                    self.handle_violation(name, tid, label, frame)
            else:
                self.timers.pop((name, tid), None)

class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret: self.frame = f
            else:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

app = Flask(__name__)
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

def gen():
    while True:
        frames = []
        if c1.frame is not None: frames.append(("Camera_1", c1.frame.copy()))
        if c2.frame is not None: frames.append(("Camera_2", c2.frame.copy()))
        
        if not frames:
            time.sleep(0.1)
            continue

        results = detect([f for _, f in frames])
        out_imgs = []
        for i, res in enumerate(results):
            name, frame = frames[i]
            monitor.process(name, res, frame)
            out_imgs.append(cv2.resize(frame, (640, 480)))

        combined = cv2.hconcat(out_imgs) if len(out_imgs) == 2 else out_imgs[0]
        _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string("<h1>Parking Monitor</h1><img src='/video_feed' width='100%'>")

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    if not os.path.exists(config.SAVE_DIR): os.makedirs(config.SAVE_DIR)
    app.run(host='0.0.0.0', port=5000, threaded=True)
