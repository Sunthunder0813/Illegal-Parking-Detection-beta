import numpy as np
import cv2
import threading
import logging
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
import config

logger = logging.getLogger("ParkingApp")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy      
        self.conf = confs     
        self.cls = clss       

class HailoDetector:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.lock = threading.Lock()
        
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape

        self.monitored_classes = [0, 2, 3, 5, 7]

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        # Identify the correct output key
        nms_keys = [k for k in raw_out.keys() if 'nms' in k.lower() or 'output' in k.lower()]
        
        if not nms_keys:
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        # FIX: Ensure we are working with a NumPy array
        detections = np.array(raw_out[nms_keys[0]])

        # If it's batched [1, N, 6], squeeze it to [N, 6]
        if len(detections.shape) == 3:
            detections = detections[0]

        for det in detections:
            # Check if detection is valid (contains coordinates, score, and class)
            if len(det) < 6: continue
            
            score = float(det[4])
            cid = int(det[5])
            
            if score > 0.1 and cid in self.monitored_classes:
                # YOLOv8 HEF format: [ymin, xmin, ymax, xmax, score, cid]
                # Convert to [xmin, ymin, xmax, ymax] for our tracker
                all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                all_confs.append(score)
                all_clss.append(cid)

        return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

    def run_detection(self, frames):
        results = []
        with self.lock:
            with self.network_group.activate():
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                    for frame in frames:
                        raw_out = pipeline.infer({self.input_info.name: self.preprocess(frame)})
                        results.append(self.postprocess(raw_out))
        return results

_detector = None
def detect(frames):
    global _detector
    if _detector is None:
        _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
