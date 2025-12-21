import numpy as np
import cv2
import logging
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
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, params)[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        logger.info(f"Hailo-8L Initialized. Input size: {self.width}x{self.height}")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        
        # Identify the NMS output tensor
        nms_node = [name for name in raw_out.keys() if 'nms' in name.lower()]
        
        if nms_node:
            detections = raw_out[nms_node[0]][0] 
            # SAFE CHECK: Ensure detections is not empty before indexing
            if detections.ndim > 1 and detections.shape[0] > 0:
                for det in detections:
                    # Hailo YOLO NMS: [ymin, xmin, ymax, xmax, score, class_id]
                    # Check if the row actually has values (score > 0)
                    score = det[4]
                    if score > 0.4: 
                        all_boxes.append([det[1], det[0], det[3], det[2]])
                        all_confs.append(score)
                        all_clss.append(det[5])

        boxes_np = np.array(all_boxes) if all_boxes else np.empty((0, 4))
        # Important: Return None for IDs if no detections, to match app.py logic
        ids_np = np.arange(len(all_boxes)) if len(all_boxes) > 0 else None
        
        return HailoResult(BoxResult(boxes_np, ids_np, np.array(all_confs), np.array(all_clss)))

    def run_detection(self, frames):
        results = []
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
    try: return detector.run_detection(frames)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return [HailoResult(BoxResult(np.empty((0,4)), None, np.array([]), np.array([])))] * len(frames)
