import numpy as np
import cv2
import logging
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

class HailoDetector:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # Configure the device interface for Raspberry Pi 5 (PCIe)
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        # Get input shape requirements from the model (usually 640x640)
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, self.channels = self.input_info.shape
        logger.info(f"Hailo Model loaded. Expected input: {self.width}x{self.height}")

    def preprocess(self, frame):
        """Resize and pad image to model input size."""
        # Resize to model dimensions
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(resized, axis=0).astype(np.float32)

    def postprocess(self, detections, threshold=0.45):
        """
        Simplistic post-processing. 
        Note: Actual HEF output format depends on how the model was compiled.
        This assumes the model includes a NMS layer.
        """
        # Dictionary to store formatted results similar to Ultralytics
        class ResultStub:
            def __init__(self, boxes, ids):
                self.boxes = self.BoxStub(boxes, ids)
            class BoxStub:
                def __init__(self, xyxy, ids):
                    self.xyxy = type('obj', (object,), {'cpu': lambda: type('obj', (object,), {'numpy': lambda: xyxy})()})()
                    self.id = type('obj', (object,), {'cpu': lambda: type('obj', (object,), {'numpy': lambda: ids})()})()

        # In a real scenario, you'd parse the output tensors here.
        # Since specific HEF output parsing is complex, we return a compatible 
        # structure for your app.py logic.
        return []

    def run_inference(self, frames):
        results = []
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate_context():
                for frame in frames:
                    input_data = {self.input_info.name: self.preprocess(frame)}
                    raw_out = infer_pipeline.infer(input_data)
                    # Process raw_out based on your specific YOLOv8 HEF architecture
                    # For this template, we return an empty list to prevent crashes
                    results.append(raw_out) 
        return results

# Singleton instance
_detector = None

def get_model():
    global _detector
    if _detector is None:
        try:
            _detector = HailoDetector(MODEL_PATH)
        except Exception as e:
            logger.error(f"Hailo Init Error: {e}")
            return None
    return _detector

def detect(frames):
    detector = get_model()
    if detector is None:
        return []
    
    try:
        # Hailo works best when processing frames sequentially in this setup
        raw_results = detector.run_inference(frames)
        
        # IMPORTANT: To make this work with your app.py logic:
        # You need a Post-Processor compatible with your specific .hef file.
        # Most pre-compiled Hailo YOLOv8 HEFs return multiple tensors (anchors).
        return raw_results 
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return []
