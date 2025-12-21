import numpy as np
import cv2
import logging
import threading
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
from config import MODEL_PATH

logger = logging.getLogger("ParkingApp")

class BoxResult:
    def __init__(self, xyxy, ids, confs, cls):
        self.xyxy = xyxy      # [N, 4]
        self.id = ids         # [N]
        self.conf = confs     # [N]
        self.cls = cls        # [N]

class HailoResult:
    def __init__(self, boxes):
        self.boxes = boxes

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
        # COCO Classes: 2:car, 3:motorcycle, 5:bus, 7:truck
        self.vehicle_classes = [2, 3, 5, 7] 

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        
        if nms_node:
            detections = raw_out[nms_node[0]]
            try:
                for i in range(len(detections[0])):
                    det = detections[0][i]
                    if len(det) >= 5:
                        score = float(det[4])
                        cid = int(det[5]) if len(det) > 5 else -1
                        if score > 0.3 and (cid in self.vehicle_classes or cid == -1):
                            # Hailo NMS: [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]
                            all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                            all_confs.append(score)
                            all_clss.append(cid)
            except Exception: pass

        boxes_np = np.array(all_boxes, dtype=np.float32) if all_boxes else np.empty((0, 4))
        return HailoResult(BoxResult(boxes_np, None, np.array(all_confs), np.array(all_clss)))

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
def get_model():
    global _detector
    if _detector is None:
        try: _detector = HailoDetector(MODEL_PATH)
        except Exception: return None
    return _detector

def detect(frames):
    detector = get_model()
    if not detector: return []
    return detector.run_detection(frames)
