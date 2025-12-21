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
        self.xyxy = xyxy
        self.id = ids
        self.conf = confs
        self.cls = cls

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
        # Vehicle classes in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7] 
        logger.info(f"Hailo-8L PCIe Active. Target Classes: {self.vehicle_classes}")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        
        if nms_node:
            detections = raw_out[nms_node[0]]
            try:
                # Iterate through the max detections (usually 80 or 100)
                for i in range(len(detections[0])):
                    det = detections[0][i]
                    if len(det) >= 5:
                        score = float(det[4])
                        class_id = int(det[5]) if len(det) > 5 else -1
                        
                        # Lowered threshold to 0.3 for better sensitivity
                        if score > 0.3 and (class_id in self.vehicle_classes or class_id == -1):
                            # Hailo NMS is [ymin, xmin, ymax, xmax] -> convert to [xmin, ymin, xmax, ymax]
                            all_boxes.append([float(det[1]), float(det[0]), float(det[3]), float(det[2])])
                            all_confs.append(score)
                            all_clss.append(class_id)
            except Exception as e:
                logger.debug(f"Parsing error: {e}")

        if all_boxes:
            logger.info(f"Detected {len(all_boxes)} vehicles")
            
        boxes_np = np.array(all_boxes, dtype=np.float32) if all_boxes else np.empty((0, 4))
        ids_np = np.arange(len(all_boxes)) if len(all_boxes) > 0 else None
        return HailoResult(BoxResult(boxes_np, ids_np, np.array(all_confs), np.array(all_clss)))

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
        except Exception as e: logger.error(f"HW Init Error: {e}"); return None
    return _detector

def detect(frames):
    detector = get_model()
    if detector is None: return []
    return detector.run_detection(frames)
