import logging
import numpy as np
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

_hailo_engine = None

class HailoInference:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=None)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)

    def run(self, frame):
        # HEF files usually expect 640x640 input
        # Note: You may need to resize/preprocess 'frame' to match model input specs
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {self.hef.get_input_vstream_infos()[0].name: np.expand_dims(frame, axis=0)}
            with self.network_group.activate_context():
                infer_results = infer_pipeline.infer(input_data)
                return infer_results

def get_model():
    global _hailo_engine
    if _hailo_engine is None:
        try:
            _hailo_engine = HailoInference(MODEL_PATH)
            logger.info(f"Successfully loaded Hailo HEF: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Could not load Hailo model. Error: {e}")
            return None
    return _hailo_engine

def detect(frames):
    engine = get_model()
    if engine is None:
        return []
    
    try:
        # Note: Hailo inference doesn't use model.track() directly.
        # You will get raw detections/tensors and must apply tracking (like ByteTrack) manually.
        results = engine.run(frames)
        return results
    except Exception as e:
        logger.error(f"Hailo Inference error: {e}")
        return []
