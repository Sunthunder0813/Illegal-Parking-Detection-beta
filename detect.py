# Classes
NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Configure which classes to detect by name here:
# Set to None for all classes, or a list of class names (e.g., ["person", "car"])
CLASSES_NAMES = list(NAMES.values()) # Example: ["person"], or None for all classes

def names_to_indices(names_list):
    if names_list is None:
        return None
    name_to_id = {v: k for k, v in NAMES.items()}
    return [name_to_id[name] for name in names_list if name in name_to_id]

CLASSES = names_to_indices(CLASSES_NAMES)

from ultralytics import YOLO
from config import MODEL_PATH

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def detect(frames):
    """
    Detect objects in a list of frames using YOLO and return results.
    Returns a list of results, one per frame.
    """
    model = get_model()
    results = model.track(
        frames,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        classes=CLASSES
    )
    return results
