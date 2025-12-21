# Camera RTSP URLs
CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.113:554/stream"

# Path to YOLO model
MODEL_PATH = "models/yolov8s.hef"

# Directory to save violation images
SAVE_DIR = "static/violations"

# Firebase configuration
DATABASE_URL = "https://illegal-parking-detectio-a8aae-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_KEY_PATH = "illegal-parking-detectio-a8aae-firebase-adminsdk-fbsvc-7181a051dc.json"

# Detection upload interval (seconds)
UPLOAD_COOLDOWN = 30

# Violation time threshold (seconds) - configurable
VIOLATION_TIME_THRESHOLD = 10  # Default: 1 minute

# Interval (seconds) to repeatedly capture/report the same violator if still present
REPEAT_CAPTURE_INTERVAL = 30  # Default: 30 seconds
