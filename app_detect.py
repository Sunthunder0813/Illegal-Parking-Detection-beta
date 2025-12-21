import numpy as np
import cv2
import logging
import threading

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
        self.xyxy = xyxy    # (N,4) [x1,y1,x2,y2]
        self.conf = conf   # (N,)
        self.cls = cls     # (N,)


class HailoDetector:
    def __init__(self, hef_path):
        self.lock = threading.Lock()

        # Load HEF
        self.hef = HEF(hef_path)
        self.device = VDevice()

        # Configure device (PCIe)
        params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, params)[0]

        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)

        # Input info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height, self.width, _ = self.input_info.shape

        # COCO classes
        # 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
        self.monitored_classes = {0, 2, 3, 5, 7}

        logger.info(
            f"Hailo YOLOv8 initialized | input={self.width}x{self.height} UINT8 RGB"
        )

    # --------------------------------------------------
    # Preprocess (MUST match HEF exactly)
    # --------------------------------------------------
    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height))
        frame = np.expand_dims(frame, axis=0)
        return frame.astype(np.uint8)

    # --------------------------------------------------
    # Postprocess (Hailo NMS output ONLY)
    # --------------------------------------------------
    def postprocess(self, raw_out):
        boxes, scores, classes = [], [], []

        # Find Hailo NMS output tensor
        nms_key = next(
            (k for k in raw_out.keys() if "nms" in k.lower()),
            None
        )

        if nms_key is None:
            logger.error("NMS output tensor not found!")
            return DetectionResult(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        detections = raw_out[nms_key].reshape(-1, 6)

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            score = float(score)
            cls_id = int(cls_id)

            # HEF already applies score >= 0.2, this is just safety
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
    # Run inference
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


# --------------------------------------------------
# Global detector instance
# --------------------------------------------------
_detector = None


def detect(frames):
    global _detector
    if _detector is None:
        _detector = HailoDetector(config.MODEL_PATH)
    return _detector.run_detection(frames)
