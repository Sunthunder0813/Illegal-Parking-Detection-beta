import numpy as np
import cv2
import threading
import logging

from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    HailoStreamInterface
)

import config

logger = logging.getLogger("ParkingApp")


class DetectionResult:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy  # pixel coords in 640x640 space
        self.conf = conf
        self.cls = cls


class HailoDetector:
    def __init__(self, hef_path):
        self.lock = threading.Lock()

        self.hef = HEF(hef_path)
        self.device = VDevice()

        params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, params)[0]

        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.h, self.w, _ = self.input_info.shape  # 640x640

        # COCO: person + vehicles
        self.monitored_classes = {0, 2, 3, 5, 7}

        logger.info("Hailo YOLOv8 detector initialized")

    # --------------------------------------------------
    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.w, self.h))
        frame = np.expand_dims(frame, axis=0)
        return frame.astype(np.uint8)

    # --------------------------------------------------
    def postprocess(self, raw_out):
        boxes, scores, classes = [], [], []

        nms_key = next((k for k in raw_out if "nms" in k.lower()), None)
        if nms_key is None:
            logger.error("NMS output not found")
            return DetectionResult(
                np.empty((0, 4)), np.array([]), np.array([])
            )

        detections = raw_out[nms_key]  # LIST (important)

        for det in detections:
            if len(det) < 6:
                continue

            x1, y1, x2, y2, score, cls_id = det
            score = float(score)
            cls_id = int(cls_id)

            if score < 0.2:
                continue

            if cls_id in self.monitored_classes:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                classes.append(cls_id)

        return DetectionResult(
            np.asarray(boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(classes, dtype=np.int32),
        )

    # --------------------------------------------------
    def run_detection(self, frames):
        results = []

        with self.lock, self.network_group.activate():
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as pipeline:

                for frame in frames:
                    raw_out = pipeline.infer(
                        {self.input_info.name: self.preprocess(frame)}
                    )
                    results.append(self.postprocess(raw_out))

        return results


_detector = None


def detect(frames):
    global _detector
    if _detector is None:
        _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
