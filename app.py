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

# COCO Class Mapping
CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# --- BYTETRACK-LITE TRACKER ---
class ByteTrackLite:
    def __init__(self):
        self.tracked_objects = {} # ID: {'box': [], 'cls': int, 'last_seen': frame}
        self.frame_count = 0
        self.next_id = 0
        # Config from your bytetrack.yaml
        self.high_thresh = 0.25
        self.low_thresh = 0.1
        self.match_thresh = 0.8 
        self.buffer = 30 

    def get_iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / float(a1 + a2 - inter + 1e-6)

    def update(self, boxes, scores, clss):
        self.frame_count += 1
        high_dets = [(b, s, c) for b, s, c in zip(boxes, scores, clss) if s >= self.high_thresh]
        low_dets = [(b, s, c) for b, s, c in zip(boxes, scores, clss) if self.low_thresh <= s < self.high_thresh]
        
        new_tracks = {}
        # Match all detections (ByteTrack two-stage approach)
        for det_box, score, cid in high_dets + low_dets:
            best_id, max_iou = None, 0.2
            for tid, tdata in self.tracked_objects.items():
                iou = self.get_iou(det_box, tdata['box'])
                if iou > max_iou: max_iou, best_id = iou, tid
            
            if best_id is not None:
                new_tracks[best_id] = {'box': det_box, 'cls': cid, 'last_seen': self.frame_count}
                if best_id in self.tracked_objects: del self.tracked_objects[best_id]
            elif score >= self.high_thresh:
                new_tracks[self.next_id] = {'box': det_box, 'cls': cid, 'last_seen': self.frame_count}
                self.next_id += 1

        # Keep buffered tracks
        for tid, tdata in self.tracked_objects.items():
            if self.frame_count - tdata['last_seen'] < self.buffer: new_tracks[tid] = tdata
        
        self.tracked_objects = new_tracks
        return {tid: d for tid, d in new_tracks.items() if d['last_seen'] == self.frame_count}

# --- FIREBASE HANDLER ---
class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
        except: self.ref = None

    def report(self, cam, v_type, frame):
        if self.ref:
            threading.Thread(target=self._upload, args=(cam, v_type, frame.copy()), daemon=True).start()

    def _upload(self, cam, v_type, frame):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{cam}_{ts}.jpg"
        cv2.imwrite(os.path.join(config.SAVE_DIR, fname), frame)
        self.ref.push({"timestamp": ts, "camera": cam, "type": v_type, "status": "Illegal Parking"})

db_client = FirebaseHandler()

# --- MONITORING LOGIC ---
class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.timers = {} # (cam, id): start_time
        self.zones = {"Camera_1": np.array([[100, 300], [1100, 300], [1100, 700], [100, 700]]), 
                      "Camera_2": np.array([[200, 200], [1000, 200], [1000, 600], [200, 600]])}

    def process(self, name, res, frame):
        h, w = frame.shape[:2]
        cv2.polylines(frame, [self.zones[name]], True, (0, 255, 0), 2)
        
        pixel_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        
        now = time.time()
        for tid, data in tracked.items():
            x1, y1, x2, y2 = map(int, data['box'])
            cid = data['cls']
            label = CLASS_NAMES.get(cid, "OBJECT")
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            # Person Detection (Visual only)
            if cid == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, f"PERSON #{tid}", (x1, y1-10), 0, 0.5, (255, 255, 0), 1)
                continue

            # Vehicle Logic
            if cv2.pointPolygonTest(self.zones[name], center, False) >= 0:
                if (name, tid) not in self.timers: self.timers[(name, tid)] = now
                dur = int(now - self.timers[(name, tid)])
                color = (0,0,255) if dur >= config.VIOLATION_TIME_THRESHOLD else (0,255,255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{tid}: {dur}s", (x1, y1-10), 0, 0.6, color, 2)
                
                if dur == config.VIOLATION_TIME_THRESHOLD:
                    db_client.report(name, label, frame)
            else:
                self.timers.pop((name, tid), None)

monitor = ParkingMonitor()
app = Flask(__name__)

# --- STREAMING ---
class Stream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret: self.frame = f
            else: time.sleep(2); self.cap = cv2.VideoCapture(self.url)

c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

def gen():
    while True:
        frames = []
        if c1.frame is not None: frames.append(("Camera_1", c1.frame.copy()))
        if c2.frame is not None: frames.append(("Camera_2", c2.frame.copy()))
        if not frames: time.sleep(0.1); continue

        results = detect([f for n, f in frames])
        out_imgs = []
        for i, res in enumerate(results):
            name, frame = frames[i]
            monitor.process(name, res, frame)
            out_imgs.append(cv2.resize(frame, (640, 480)))

        combined = cv2.hconcat(out_imgs) if len(out_imgs)==2 else out_imgs[0]
        _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return render_template_string("<h1>Parking & Person Monitor</h1><img src='/video_feed' width='100%'>")

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, threaded=True)
