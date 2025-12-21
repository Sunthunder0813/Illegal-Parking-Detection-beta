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

# Violation Logic
VIOLATION_TIME_THRESHOLD = 10  # Seconds before it's a violation
REPEAT_CAPTURE_INTERVAL = 30   # Cooldown between logs for same car

# --- THE MISSING LINE ---
DETECTION_THRESHOLD = 0.3      # Minimum confidence (0.0 to 1.0)
