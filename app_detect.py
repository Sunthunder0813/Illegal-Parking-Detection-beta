import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH

# Specific COCO classes for parking detection
NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Pre-calculate the indices for the 'classes' argument in track()
CLASSES = list(NAMES.keys())

_model = None

def get_model():
    global _model
    if _model is None:
        # Load the model. Ensure config.MODEL_PATH points to 'yolo11s.hef'
        _model = YOLO(MODEL_PATH) 
    return _model

def detect(frames):
    """
    Detects and tracks vehicles using Hailo-8L.
    Accepts a list of frames (for dual camera support).
    """
    model = get_model()
    
    # .track handles persistent ID assignment across frames
    results = model.track(
        frames,
        persist=True,      # Required to keep the same ID for a parked car
        tracker="bytetrack.yaml", 
        verbose=False,
        classes=CLASSES    # Only detect vehicles (car, motorcycle, bus, truck)
    )
    return results
