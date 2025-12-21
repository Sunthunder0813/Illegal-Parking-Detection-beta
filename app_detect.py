import numpy as np
import cv2
import logging
import threading
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
import config

logger = logging.getLogger("ParkingApp")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy      # [N, 4]
        self.conf = confs     # [N]
        self.cls = clss       # [N]

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
        self.vehicle_classes = [2, 3, 5, 7] # COCO: car, motorcycle, bus, truck

    def preprocess(self, frame):
        return np.expand_dims(cv2.resize(frame, (self.width, self.height)), axis=0)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        if nms_node:
            detections = raw_out[nms_node[0]]
            for det in detections[0]:
                if len(det) >= 5 and det[4] > 0.1: # ByteTrack needs low scores
                    cid = int(det[5])
                    if cid in self.vehicle_classes:
                        all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                        all_confs.append(float(det[4]))
                        all_clss.append(cid)
        return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

    def run_detection(self, frames):
        results = []
        with self.lock, self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                for frame in frames:
                    raw_out = pipeline.infer({self.input_info.name: self.preprocess(frame)})
                    results.append(self.postprocess(raw_out))
        return results

_detector = None
def detect(frames):
    global _detector
    if _detector is None: _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
