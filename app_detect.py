import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH

# Classes mapping
NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Configure which classes to detect
CLASSES_NAMES = list(NAMES.values())

def names_to_indices(names_list):
    if names_list is None:
        return None
    name_to_id = {v: k for k, v in NAMES.items()}
    return [name_to_id[name] for name in names_list if name in name_to_id]

CLASSES = names_to_indices(CLASSES_NAMES)

_model = None

def get_model():
    global _model
    if _model is None:
        # Load the model. To use Hailo, the MODEL_PATH must point to a .hef file
        # Ultralytics handles the Hailo offloading if the .hef path is provided
        _model = YOLO(MODEL_PATH) 
    return _model

def detect(frames):
    """
    Detect objects using Hailo-8L NPU acceleration via Ultralytics.
    """
    model = get_model()
    
    # .track() will use the Hailo chip if the model loaded is a .hef
    results = model.track(
        frames,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        classes=CLASSES
    )
    return results
