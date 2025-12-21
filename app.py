import cv2
import threading
import time
import os
import datetime
import numpy as np
from flask import Flask, Response, render_template_string
import firebase_admin
from firebase_admin import credentials, db
from app_detect import detect
import config

# COCO Class Mapping
CLASS_NAMES = {2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

class ByteTrackLite:
    def __init__(self):
        self.tracked_objects = {} # ID: {'box': [], 'cls': int, 'last_seen': frame}
        self.frame_count = 0
        self.next_id = 0
        self.high_thresh, self.low_thresh, self.buffer = 0.25, 0.1, 30

    def get_iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1, a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / float(a1 + a2 - inter + 1e-6)

    def update(self, boxes, scores, clss):
        self.frame_count += 1
        high_dets = [(b, s, c) for b, s, c in zip(boxes, scores, clss) if s >= self.high_thresh]
        low_dets = [(b, s, c) for b, s, c in zip(boxes, scores, clss) if self.low_thresh <= s < self.high_thresh]
        
        new_tracks = {}
        # Match High & Low detections (ByteTrack logic)
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

        for tid, tdata in self.tracked_objects.items():
            if self.frame_count - tdata['last_seen'] < self.buffer: new_tracks[tid] = tdata
        
        self.tracked_objects = new_tracks
        return {tid: d for tid, d in new_tracks.items() if d['last_seen'] == self.frame_count}

class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
        except: self.ref = None

    def report(self, cam, vehicle_type, frame):
        if self.ref:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{cam}_{ts}.jpg"
            cv2.imwrite(os.path.join(config.SAVE_DIR, fname), frame)
            self.ref.push({"timestamp": ts, "camera": cam, "type": vehicle_type, "status": "Illegal Parking"})

db_client = FirebaseHandler()

class Monitor:
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
            v_type = CLASS_NAMES.get(data['cls'], "VEHICLE")
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            if cv2.pointPolygonTest(self.zones[name], center, False) >= 0:
                if (name, tid) not in self.timers: self.timers[(name, tid)] = now
                dur = int(now - self.timers[(name, tid)])
                color = (0,0,255) if dur > config.VIOLATION_TIME_THRESHOLD else (0,255,255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Display Label: "TRUCK #4: 15s"
                cv2.putText(frame, f"{v_type} #{tid}: {dur}s", (x1, y1-10), 0, 0.6, color, 2)
                
                if dur == config.VIOLATION_TIME_THRESHOLD:
                    db_client.report(name, v_type, frame)
            else:
                self.timers.pop((name, tid), None)

parking_monitor = Monitor()
app = Flask(__name__)

# RTSP Streamer Class
class Stream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret: self.frame = f
            else: time.sleep(1)

cam1, cam2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

def gen():
    while True:
        cams = []
        if cam1.frame is not None: cams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: cams.append(("Camera_2", cam2.frame.copy()))
        if not cams: time.sleep(0.1); continue

        results = detect([f for n, f in cams])
        imgs = []
        for i, res in enumerate(results):
            name, frame = cams[i]
            parking_monitor.process(name, res, frame)
            imgs.append(cv2.resize(frame, (640, 480)))

        combined = cv2.hconcat(imgs) if len(imgs)==2 else imgs[0]
        _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): return "<h1>Parking Monitor</h1><img src='/video_feed' width='100%'>"

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
