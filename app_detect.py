import numpy as np
import cv2
import threading
import logging
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
import config

logger = logging.getLogger("ParkingApp")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy      # [[xmin, ymin, xmax, ymax], ...]
        self.conf = confs     # [score, ...]
        self.cls = clss       # [class_id, ...]

class HailoDetector:
    def __init__(self, hef_path):
        self.target = VDevice()
        self.hef = HEF(hef_path)
        self.lock = threading.Lock()
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        # Get input info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        self.monitored_classes = [0, 2, 3, 5, 7]

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return {self.input_info.name: np.expand_dims(resized, axis=0).astype(np.uint8)}

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        # Find the NMS/Output key
        nms_key = [k for k in raw_out.keys() if 'nms' in k.lower() or 'output' in k.lower()]
        if not nms_key:
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        detections = raw_out[nms_key[0]]
        
        # Ensure detections is a 2D array [N, 6]
        if len(detections.shape) == 3:
            detections = detections[0]

        for det in detections:
            if len(det) < 6: continue
            score = float(det[4])
            cid = int(det[5])
            
            if score > 0.1 and cid in self.monitored_classes:
                # YOLOv8 Hailo Output: [ymin, xmin, ymax, xmax, score, cid]
                # Converting to tracker format: [xmin, ymin, xmax, ymax]
                all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                all_confs.append(score)
                all_clss.append(cid)

        return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

    def run_detection(self, frames):
        results = []
        with self.lock:
            # Re-activating the group and creating the stream context per batch 
            # to avoid the 'AttributeError' found in persistent pipelines
            with self.network_group.activate():
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                    for frame in frames:
                        data = self.preprocess(frame)
                        raw_out = pipeline.infer(data)
                        results.append(self.postprocess(raw_out))
        return results

_detector = None
def detect(frames):
    global _detector
    if _detector is None:
        _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
