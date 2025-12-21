import numpy as np
import cv2
import logging
import threading
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

class BoxResult:
    def __init__(self, xyxy, confs, cls):
        self.xyxy = xyxy      # [N, 4]
        self.conf = confs     # [N]
        self.cls = cls        # [N]
        self.id = None        # Placeholder for tracker

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
        self.vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        
        if nms_node:
            detections = raw_out[nms_node[0]]
            try:
                for det in detections[0]:
                    if len(det) >= 5:
                        score = float(det[4])
                        cid = int(det[5]) if len(det) > 5 else -1
                        # We use 0.1 threshold to allow ByteTrack to see "weak" detections
                        if score > 0.1 and (cid in self.vehicle_classes or cid == -1):
                            # Scale to 0-1 range
                            all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                            all_confs.append(score)
                            all_clss.append(cid)
            except: pass
        return BoxResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

    def run_detection(self, frames):
        results = []
        with self.lock:
            with self.network_group.activate():
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                    for frame in frames:
                        input_data = {self.input_info.name: self.preprocess(frame)}
                        raw_out = pipeline.infer(input_data)
                        results.append(self.postprocess(raw_out))
        return results

_detector = None
def detect(frames):
    global _detector
    if _detector is None: _detector = HailoDetector(MODEL_PATH)
    return _detector.run_detection(frames)
