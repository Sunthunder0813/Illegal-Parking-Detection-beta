import numpy as np
import cv2
import logging
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

# Wrapper classes to mimic the Ultralytics 'Results' structure
class BoxResult:
    def __init__(self, xyxy, ids, confs, cls):
        self.xyxy = xyxy      # numpy array [N, 4] (Normalized 0-1)
        self.id = ids         # numpy array [N]
        self.conf = confs     # numpy array [N]
        self.cls = cls        # numpy array [N]

class HailoResult:
    def __init__(self, boxes):
        self.boxes = boxes

class HailoDetector:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # Configure for Raspberry Pi 5 PCIe
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        # Get model input requirements
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        logger.info(f"Hailo-8L Initialized. Model expected input: {self.width}x{self.height}")

    def preprocess(self, frame):
        """Resize frame to model dimensions."""
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, raw_out):
        """Parse raw tensors into HailoResult objects."""
        all_boxes, all_confs, all_clss = [], [], []
        
        # Find the NMS output layer (usually contains 'nms' in the name)
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        
        if nms_node:
            detections = raw_out[nms_node[0]][0] 
            for det in detections:
                # Hailo YOLOv8 NMS format: [ymin, xmin, ymax, xmax, score, class_id]
                if det[4] > 0.4: # Confidence Threshold
                    # Reorder to [xmin, ymin, xmax, ymax]
                    all_boxes.append([det[1], det[0], det[3], det[2]])
                    all_confs.append(det[4])
                    all_clss.append(det[5])

        boxes_np = np.array(all_boxes) if all_boxes else np.empty((0, 4))
        # Use simple indices as IDs for initial compatibility
        ids_np = np.arange(len(all_boxes)) if all_boxes else None
        
        return HailoResult(BoxResult(boxes_np, ids_np, np.array(all_confs), np.array(all_clss)))

    def run_detection(self, frames):
        results = []
        # Critical Fix: Use .activate() and keep the pipeline inside the context
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                for frame in frames:
                    input_data = {self.input_info.name: self.preprocess(frame)}
                    raw_out = pipeline.infer(input_data)
                    results.append(self.postprocess(raw_out))
        return results

# Singleton management
_detector = None

def get_model():
    global _detector
    if _detector is None:
        try:
            _detector = HailoDetector(MODEL_PATH)
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return None
    return _detector

def detect(frames):
    detector = get_model()
    if detector is None: return []
    try:
        return detector.run_detection(frames)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return []
