import cv2
import threading
import time
import torch
from ultralytics import YOLO
from detect import NAMES, CLASSES
from flask import Flask, Response, render_template

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/yolov8n.pt" 
INPUT_SIZE = 416
CONF = 0.4
# tracker options: "botsort.yaml" or "bytetrack.yaml"
# ByteTrack is usually faster for high-density overlapping
TRACKER_TYPE = "models/bytetrack.yaml" 

CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.113:554/stream"

# =============================
# THREADED CAMERA CLASS
# =============================
class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = False, None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.status, self.frame = True, frame
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# =============================
# INITIALIZE
# =============================
model = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    model.to('cuda').half()

cam1 = RTSPStream(CAM1_URL)
cam2 = RTSPStream(CAM2_URL)

app = Flask(__name__)

# =============================
# VIDEO GENERATOR
# =============================
def gen_frames():
    prev_time = time.time()
    while True:
        f1, f2 = cam1.read(), cam2.read()
        if f1 is None or f2 is None:
            time.sleep(0.01)
            continue

        results = model.track(
            [f1, f2],
            imgsz=INPUT_SIZE,
            conf=CONF,
            classes=CLASSES,
            persist=True,
            tracker=TRACKER_TYPE,
            verbose=False
        )

        out1 = results[0].plot()
        out2 = results[1].plot()
        combined = cv2.hconcat([out1, out2])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(combined, f"FPS: {fps:.1f} (Tracking Active)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', combined)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =============================
# FLASK ROUTES
# =============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =============================
# MAIN ENTRY
# =============================
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cam1.stop()
        cam2.stop()
        cv2.destroyAllWindows()