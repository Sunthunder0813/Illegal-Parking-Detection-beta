import numpy as np
import cv2
from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams
from config import MODEL_PATH # Ensure this points to a .hef file now

NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class HailoDetector:
    def __init__(self, hef_path):
        self.target = VDevice()
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(self.hef, interface=self.target.get_default_interface())
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=FormatType.FLOAT32)
        
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape

    def preprocess(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def detect(self, frames):
        results = []
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            for frame in frames:
                if frame is None: continue
                prep_frame = self.preprocess(frame)
                input_data = {self.input_info.name: np.expand_dims(prep_frame, axis=0)}
                
                # Inference on Hailo-8L
                raw_out = infer_pipeline.infer(input_data)
                
                # Note: Real-world Hailo output requires NMS post-processing.
                # Here we return a mock-structure compatible with your tracker logic.
                results.append(self.dummy_post_process(raw_out, frame.shape))
        return results

    def dummy_post_process(self, raw_out, shape):
        # In a real setup, use hailo_model_zoo.core.postprocessing
        # This is a placeholder object to keep your main loop running
        class Box:
            def __init__(self):
                self.xyxy = [np.array([0,0,0,0])]
                self.cls = [2]
                self.id = [None]
        class Res:
            def __init__(self): self.boxes = []
            def plot(self): return np.zeros(shape, dtype=np.uint8)
        return Res()

_detector = None
def detect(frames):
    global _detector
    if _detector is None:
        # Ensure MODEL_PATH in config.py points to a .hef file
        _detector = HailoDetector(MODEL_PATH.replace(".pt", ".hef"))
    return _detector.detect(frames)