import numpy as np
import cv2
import logging
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

class HailoDetector:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # Configure for Raspberry Pi 5 PCIe
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Prepare stream parameters
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        # Input info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        
        logger.info(f"Hailo-8L Initialized. Model: {hef_path}")

    def preprocess(self, frame):
        """Resize frame to match model input dimensions."""
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def run_detection(self, frames):
        results = []
        # The key fix: Wrap the activation and the inference in the same 'with' block
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                for frame in frames:
                    input_data = {self.input_info.name: self.preprocess(frame)}
                    raw_out = infer_pipeline.infer(input_data)
                    
                    # We return raw_out. You may need a post-processing step 
                    # here to convert tensors to boxes [x, y, w, h].
                    results.append(raw_out)
        return results

# Singleton logic
_detector = None

def get_model():
    global _detector
    if _detector is None:
        try:
            _detector = HailoDetector(MODEL_PATH)
        except Exception as e:
            logger.error(f"Hardware Error: {e}")
            return None
    return _detector

def detect(frames):
    detector = get_model()
    if detector is None:
        return []
    
    try:
        # Pass the frames to the detector
        return detector.run_detection(frames)
    except Exception as e:
        # This catches the 'Network group not activated' error if it persists
        logger.error(f"Inference error: {e}")
        return []
