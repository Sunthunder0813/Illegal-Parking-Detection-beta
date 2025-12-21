import cv2
import threading
import time
import os
import datetime
import numpy as np
from flask import Flask, Response
from app_detect import detect
from config import (CAM1_URL, CAM2_URL, SAVE_DIR, VIOLATION_TIME_THRESHOLD, REPEAT_CAPTURE_INTERVAL)

# =============================
# DEBUG LOGGING & SETUP
# =============================
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

os.makedirs(SAVE_DIR, exist_ok=True)
# Adjust these zones based on your camera view
ZONE_CAM1 = np.array([[50, 200], [600, 200], [600, 450], [50, 450]], np.int32)
ZONE_CAM2 = np.array([[50, 200], [600, 200], [600, 450], [50, 450]], np.int32)

class VehicleTracker:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_seen = {} # {cam: {id: time}}
        self.violated_ids = {}
        self.zones = {"Camera_1": ZONE_CAM1, "Camera_2": ZONE_CAM2}

    def is_box_in_zone(self, box_xyxy, cam_name):
        zone = self.zones.get(cam_name)
        if zone is None: return False
        x1, y1, x2, y2 = box_xyxy
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return cv2.pointPolygonTest(zone, center, False) >= 0

    def update(self, cam_name, detections):
        """
        detections: list of (id, box_xyxy)
        returns: (new_violations_count, list_of_violator_ids)
        """
        now = time.time()
        if cam_name not in self.first_seen:
            self.first_seen[cam_name], self.violated_ids[cam_name] = {}, set()

        new_count = 0
        current_violators = []

        for vid, box in detections:
            if self.is_box_in_zone(box, cam_name):
                if vid not in self.first_seen[cam_name]:
                    self.first_seen[cam_name][vid] = now
                
                duration = now - self.first_seen[cam_name][vid]
                if vid not in self.violated_ids[cam_name] and duration >= self.threshold:
                    self.violated_ids[cam_name].add(vid)
                    new_count += 1
                    current_violators.append(vid)
            else:
                self.first_seen[cam_name].pop(vid, None)
        
        return new_count, current_violators

tracker = VehicleTracker(VIOLATION_TIME_THRESHOLD)

class RTSPStream:
    def __init__(self, url, name):
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.frame = frame
            else: time.sleep(0.01)

cam1 = RTSPStream(CAM1_URL, "Camera_1")
cam2 = RTSPStream(CAM2_URL, "Camera_2")
app = Flask(__name__)

def gen_frames():
    while True:
        # 1. Grab Frames
        active_streams = []
        if cam1.frame is not None: active_streams.append(("Camera_1", cam1.frame.copy()))
        if cam2.frame is not None: active_streams.append(("Camera_2", cam2.frame.copy()))
        
        if not active_streams:
            continue

        # 2. Hailo Inference
        frames_to_detect = [f for n, f in active_streams]
        results = detect(frames_to_detect) # Returns list of Res objects

        display_frames = []
        for i, res in enumerate(results):
            cam_name = active_streams[i][0]
            frame = active_streams[i][1]
            
            # Map Hailo Results to Tracker
            # (Note: Since Hailo-8L doesn't have built-in tracking in basic scripts, 
            # we use coordinates as temporary IDs or a simple distance tracker)
            formatted_detections = []
            for box in res.boxes:
                # In a real setup, box.id is provided by a tracker like ByteTrack
                # For debug, we use the class name as a dummy ID
                formatted_detections.append((box.cls[0], box.xyxy[0]))

            # 3. Update Logic
            new_v, violators = tracker.update(cam_name, formatted_detections)
            if new_v > 0:
                logger.info(f"VIOLATION DETECTED on {cam_name}")

            # 4. Drawing for Debugging
            cv2.polylines(frame, [tracker.zones[cam_name]], True, (0, 255, 0), 2)
            for vid, box in formatted_detections:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if vid in violators else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{vid}", (x1, y1-10), 0, 0.6, color, 2)

            display_frames.append(cv2.resize(frame, (640, 480)))

        # 5. Merge and Stream
        final_view = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
        _, buffer = cv2.imencode('.jpg', final_view)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
