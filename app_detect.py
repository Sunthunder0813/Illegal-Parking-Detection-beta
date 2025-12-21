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
        
        # FIX 1: Use .activate() instead of .activate_context()
        self.active_network = self.network_group.activate()
        
        # Prepare stream parameters
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        # Input info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        
        logger.info(f"Hailo-8L Ready. Input size: {self.width}x{self.height}")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def detect_frames(self, frames):
        results = []
        # FIX 2: Keep the pipeline open for the duration of the detection call
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            for frame in frames:
                input_data = {self.input_info.name: self.preprocess(frame)}
                raw_out = infer_pipeline.infer(input_data)
                
                # Mock result object to keep your app logic working
                # We return the raw output for now; you may need to parse 
                # boxes depending on your specific HEF.
                results.append(raw_out)
        return results

    def __del__(self):
        # Clean up hardware resources on shutdown
        if hasattr(self, 'active_network'):
            self.active_network.close()

# Singleton instance
_detector = None

def get_model():
    global _detector
    if _detector is None:
        try:
            _detector = HailoDetector(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize Hailo hardware: {e}")
            return None
    return _detector

def detect(frames):
    detector = get_model()
    if detector is None:
        return []
    
    try:
        return detector.detect_frames(frames)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return []
