import numpy as np
import cv2
import threading
import logging
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, HailoStreamInterface)
import config

logger = logging.getLogger("ParkingApp")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy        # [N, 4] array of bounding boxes xmin,ymin,xmax,ymax normalized 0-1
        self.conf = confs       # [N] array of confidence scores
        self.cls = clss         # [N] array of class IDs

class HailoDetector:
    def __init__(self, hef_path):
        logger.info(f"Loading HEF from: {hef_path}")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.lock = threading.Lock()
        
        # Configure PCIe interface
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.target.configure(self.hef, params)
        if not network_groups:
            raise RuntimeError("No network groups found in HEF")
        self.network_group = network_groups[0]
        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape
        
        # COCO IDs to monitor: Person=0, Car=2, Motorcycle=3, Bus=5, Truck=7
        self.monitored_classes = [0, 2, 3, 5, 7]

    def preprocess(self, frame):
        # Resize frame to expected input size of the network
        resized = cv2.resize(frame, (self.width, self.height))
        # Hailo expects uint8 images in NHWC format, no normalization
        input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
        return input_tensor

    def postprocess(self, raw_out):
        all_boxes = []
        all_confs = []
        all_clss = []

        # Find the NMS output tensor by key heuristics
        nms_key = None
        for key in raw_out.keys():
            if 'nms' in key.lower() or 'output' in key.lower():
                nms_key = key
                break

        if nms_key is None:
            logger.warning("No NMS output found in raw_out keys")
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        detections = raw_out[nms_key]
        detections = np.array(detections)  # Ensure numpy array

        # Expected shape: [1, 100, 6] where each detection: [ymin, xmin, ymax, xmax, score, class_id]
        if detections.ndim != 3 or detections.shape[2] < 6:
            logger.warning(f"Unexpected detection output shape: {detections.shape}")
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        for det in detections[0]:
            score = float(det[4])
            if score < 0.2:  # confidence threshold
                continue
            cls_id = int(det[5])
            if cls_id not in self.monitored_classes:
                continue
            ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
            box = [xmin, ymin, xmax, ymax]
            all_boxes.append(box)
            all_confs.append(score)
            all_clss.append(cls_id)

        return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

    def run_detection(self, frames):
        results = []
        with self.lock, self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                for frame in frames:
                    input_tensor = self.preprocess(frame)
                    raw_out = pipeline.infer({self.input_info.name: input_tensor})

                    # Debug prints to verify model output
                    logger.debug(f"Raw output keys: {list(raw_out.keys())}")
                    for k, v in raw_out.items():
                        logger.debug(f"Output '{k}': shape={np.array(v).shape}, dtype={np.array(v).dtype}, min={np.min(v)}, max={np.max(v)}")

                    result = self.postprocess(raw_out)
                    logger.debug(f"Detections found: {len(result.xyxy)}")
                    results.append(result)
        return results

_detector = None
def detect(frames):
    global _detector
    if _detector is None:
        logger.info("Initializing HailoDetector")
        _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
