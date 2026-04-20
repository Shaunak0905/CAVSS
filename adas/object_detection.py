"""
adas/object_detection.py
YOLOv8-nano object detection wrapper for road-facing ADAS feed.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# COCO class IDs relevant for driving
_DRIVING_CLASSES: Dict[int, str] = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


@dataclass
class Detection:
    """A single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    distance_estimate: Optional[float] = None  # metres (approximate)
    ttc: Optional[float] = None                # time to collision


class ObjectDetector:
    """
    YOLOv8-nano detector for road scene objects.

    On first instantiation, ultralytics will auto-download yolov8n.pt
    if it is not present locally.

    Args:
        model_path: Path to YOLOv8 weights file.
        confidence_threshold: Minimum detection confidence.
        iou_threshold: NMS IOU threshold.
        device: "cpu", "cuda", or "mps".
        target_classes: COCO class IDs to keep (None = all).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        target_classes: Optional[List[int]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes or list(_DRIVING_CLASSES.keys())

        # Lazy import so the rest of the system loads even if ultralytics is absent
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(model_path)
            logger.info(f"YOLOv8 loaded: {model_path} on {device}")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            self._model = None
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            self._model = None

        # FPS tracking
        self._frame_times: List[float] = []
        self._max_time_samples = 30

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: BGR image (e.g. 1280×720 from YouTubeFeed).

        Returns:
            (detections, processing_time_ms)
        """
        t0 = time.perf_counter()

        if self._model is None:
            return [], 0.0

        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                classes=self.target_classes,
            )[0]
        except Exception as exc:
            logger.error(f"YOLO inference error: {exc}")
            return [], 0.0

        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _DRIVING_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=_DRIVING_CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    area=area,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Track FPS
        self._frame_times.append(time.time())
        if len(self._frame_times) > self._max_time_samples:
            self._frame_times.pop(0)

        return detections, elapsed_ms

    @property
    def fps_actual(self) -> float:
        """Measured inference FPS."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame.

        Args:
            frame: BGR image.
            detections: List from detect().

        Returns:
            Annotated copy of the frame.
        """
        out = frame.copy()
        colours: Dict[str, Tuple[int, int, int]] = {
            "Person": (0, 0, 255),
            "Bicycle": (255, 165, 0),
            "Car": (0, 255, 0),
            "Motorcycle": (0, 165, 255),
            "Bus": (255, 0, 255),
            "Truck": (255, 0, 0),
        }
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = colours.get(det.class_name, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            if det.distance_estimate is not None:
                label += f" {det.distance_estimate:.1f}m"
            if det.ttc is not None:
                label += f" TTC:{det.ttc:.1f}s"
            cv2.putText(
                out, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )
        return out
