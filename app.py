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

# --- BYTETRACK-INSPIRED TRACKER ---
class ByteTrackLite:
    def __init__(self):
        self.tracked_objects = {}  # ID: {'box': [], 'last_seen': frame, 'active': bool}
        self.frame_count = 0
        self.next_id = 0
        
        # Parameters from your bytetrack.yaml
        self.high_thresh = 0.25
        self.low_thresh = 0.1
        self.match_thresh = 0.8  # Similarity threshold
        self.buffer = 30         # Frames to keep lost tracks alive

    def get_iou(self, b1, b2):
        xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / float(area1 + area2 - inter + 1e-6)

    def update(self, boxes, scores):
        self.frame_count += 1
        # Split detections based on yaml thresholds
        high_dets = [(b, s) for b, s in zip(boxes, scores) if s >= self.high_thresh]
        low_dets = [(b, s) for b, s in zip(boxes, scores) if self.low_thresh <= s < self.high_thresh]
        
        updated_tracks = {}
        
        # 1. Match high-score detections to existing tracks
        for det_box, score in high_dets:
            best_id = None; max_iou = 0
            for tid, tdata in self.tracked_objects.items():
                iou = self.get_iou(det_box, tdata['box'])
                if iou > max_iou: max_iou = iou; best_id = tid
            
            if best_id is not None and max_iou > (1 - self.match_thresh):
                updated_tracks[best_id] = {'box': det_box, 'last_seen': self.frame_count}
                if best_id in self.tracked_objects: del self.tracked_objects[best_id]
            else:
                updated_tracks[self.next_id] = {'box': det_box, 'last_seen': self.frame_count}
                self.next_id += 1

        # 2. Match low-score detections to remaining (potentially occluded) tracks
        for det_box, score in low_dets:
            best_id = None; max_iou = 0
            for tid, tdata in self.tracked_objects.items():
                iou = self.get_iou(det_box, tdata['box'])
                if iou > max_iou: max_iou = iou; best_id = tid
            
            if best_id is not None and max_iou > (1 - self.match_thresh):
                updated_tracks[best_id] = {'box': det_box, 'last_seen': self.frame_count}
                del self.tracked_objects[best_id]

        # 3. Handle 'lost' tracks (keep in buffer)
        for tid, tdata in self.tracked_objects.items():
            if self.frame_count - tdata['last_seen'] < self.buffer:
                updated_tracks[tid] = tdata
        
        self.tracked_objects = updated_tracks
        # Only return objects visible in the CURRENT frame for display
        return {tid: d['box'] for tid, d in updated_tracks.items() if d['last_seen'] == self.frame_count}

# --- PARKING LOGIC ---
ZONE_CAM1 = np.array([[100, 300], [1100, 300], [1100, 700], [100, 700]], np.int32)
ZONE_CAM2 = np.array([[200, 200], [1000, 200], [1000, 600], [200, 600]], np.int32)

class FirebaseHandler:
    def __init__(self):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {'databaseURL': config.DATABASE_URL})
            self.ref = db.reference('violations_history')
            logger.info("Firebase connected.")
        except Exception as e:
            logger.error(f"Firebase failed: {e}"); self.ref = None

    def report(self, cam_name, frame):
        if self.ref:
            threading.Thread(target=self._upload, args=(cam_name, frame.copy()), daemon=True).start()

    def _upload(self, cam_name, frame):
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{cam_name}_{ts}.jpg"
            cv2.imwrite(os.path.join(config.SAVE_DIR, fname), frame)
            self.ref.push({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cam_name, "status": "Illegal Parking", "image": fname
            })
        except Exception as e: logger.error(f"Upload error: {e}")

db_client = FirebaseHandler()

class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.park_start_times = {} # (cam, id): timestamp
        self.last_upload_time = {} # (cam, id): timestamp
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}

    def process(self, cam_name, res, frame):
        h, w = frame.shape[:2]
        cv2.polylines(frame, [self.zones[cam_name]], True, (0, 255, 0), 3)
        
        pixel_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in res.xyxy]
        tracked_objects = self.trackers[cam_name].update(pixel_boxes, res.conf)
        
        now = time.time()
        for tid, box in tracked_objects.items():
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            if cv2.pointPolygonTest(self.zones[cam_name], center, False) >= 0:
                key = (cam_name, tid)
                if key not in self.park_start_times:
                    self.park_start_times[key] = now
                
                duration = now - self.park_start_times[key]
                color = (0,0,255) if duration > config.VIOLATION_TIME_THRESHOLD else (0,255,255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"VEHICLE {tid}: {int(duration)}s", (x1, y1-10), 0, 0.6, color, 2)
                
                # Report violation
                if duration > config.VIOLATION_TIME_THRESHOLD:
                    last_up = self.last_upload_time.get(key, 0)
                    if now - last_up > config.REPEAT_CAPTURE_INTERVAL:
                        db_client.report(cam_name, frame)
                        self.last_upload_time[key] = now
            else:
                self.park_start_times.pop((cam_name, tid), None)

parking_monitor = ParkingMonitor()

# --- STREAMING ---
class RTSPStream:
    def __init__(self, url, name):
        self.url, self.name, self.frame = url, name, None
        self.cap = cv2.VideoCapture(url)
        threading.Thread(target=self._update, daemon=True).start()
    def _update(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: time.sleep(2); self.cap = cv2.VideoCapture(self.url)

cam1 = RTSPStream(config.CAM1_URL, "Camera_1")
cam2 = RTSPStream(config.CAM2_URL, "Camera_2")
app = Flask(__name__)

def gen_frames():
    while True:
        streams = []
        if cam1.frame is not None: streams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: streams.append(("Camera_2", cam2.frame.copy()))
        if not streams: time.sleep(0.1); continue

        results = detect([f for n, f in streams])
        
        output_frames = []
        for i, res in enumerate(results):
            name, frame = streams[i]
            parking_monitor.process(name, res, frame)
            output_frames.append(cv2.resize(frame, (640, 480)))

        # Create side-by-side view
        if len(output_frames) == 2:
            combined = cv2.hconcat(output_frames)
        else:
            combined = output_frames[0]

        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("<h1>Hailo-8L ByteTrack Monitor</h1><img src='/video_feed' width='100%'>")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
