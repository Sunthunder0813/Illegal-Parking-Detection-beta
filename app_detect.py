import logging
from ultralytics import YOLO
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            # Try loading the model
            _model = YOLO(MODEL_PATH)
            logger.info(f"Successfully loaded model: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Could not load YOLO model at {MODEL_PATH}. Check if file is corrupted. Error: {e}")
            return None
    return _model

def detect(frames):
    model = get_model()
    if model is None:
        # If model failed to load, return empty results so the app keeps running
        return []
    
    try:
        results = model.track(
            frames,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=[2, 3, 5, 7]
        )
        return results
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return []
