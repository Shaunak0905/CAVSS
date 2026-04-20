"""
adas/lane_detection.py
Classical CV lane detection: Canny edges → Hough lines → lane averaging.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class LaneResult:
    """Detected lane geometry and departure assessment."""
    left_lane: Optional[Tuple[int, int, int, int]]   # x1,y1,x2,y2
    right_lane: Optional[Tuple[int, int, int, int]]
    lane_center: int                                  # pixel x of lane centre
    vehicle_offset: float                             # -1..1, negative = left
    departure_warning: bool
    departure_risk: float                             # 0–1 for CRE


class LaneDetector:
    """
    Detects left/right lane markings in a road-facing frame.

    Pipeline:
        grey → gaussian blur → canny → ROI mask → hough → classify
        → average/extrapolate → smooth over N frames

    Args:
        config: Dict from config.yaml['adas']['lane_detection'].
        frame_shape: (height, width) of the video feed.
    """

    def __init__(self, config: dict, frame_shape: Tuple[int, int] = (720, 1280)) -> None:
        self._h, self._w = frame_shape

        # Canny
        self._canny_lo: int = config.get("canny_low_threshold", 50)
        self._canny_hi: int = config.get("canny_high_threshold", 150)

        # Hough
        self._rho: int = config.get("hough_rho", 2)
        self._theta: float = np.deg2rad(config.get("hough_theta_degrees", 1))
        self._hough_thr: int = config.get("hough_threshold", 50)
        self._min_line: int = config.get("hough_min_line_length", 40)
        self._max_gap: int = config.get("hough_max_line_gap", 100)

        # ROI (fraction of frame)
        self._roi_top: float = config.get("roi_top", 0.55)
        self._roi_bottom: float = config.get("roi_bottom", 0.95)
        self._roi_left: float = config.get("roi_left", 0.1)
        self._roi_right: float = config.get("roi_right", 0.9)

        # Departure
        self._departure_thr: float = config.get("departure_threshold", 0.15)

        # Smoothing
        smooth_frames: int = config.get("lane_smoothing_frames", 5)
        self._left_history: deque = deque(maxlen=smooth_frames)
        self._right_history: deque = deque(maxlen=smooth_frames)

        self._roi_vertices = self._build_roi()
        logger.info(f"LaneDetector ready ({self._w}×{self._h})")

    def _build_roi(self) -> np.ndarray:
        h, w = self._h, self._w
        top_y = int(h * self._roi_top)
        bot_y = int(h * self._roi_bottom)
        left_x = int(w * self._roi_left)
        right_x = int(w * self._roi_right)
        mid_left = int(w * 0.40)
        mid_right = int(w * 0.60)
        return np.array([[
            (left_x, bot_y),
            (mid_left, top_y),
            (mid_right, top_y),
            (right_x, bot_y),
        ]], dtype=np.int32)

    def update_frame_shape(self, h: int, w: int) -> None:
        """Rebuild ROI when frame dimensions change."""
        if (h, w) != (self._h, self._w):
            self._h, self._w = h, w
            self._roi_vertices = self._build_roi()

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Detect lane markings in a single BGR frame.

        Args:
            frame: BGR road-facing image.

        Returns:
            LaneResult with lane coordinates and departure info.
        """
        h, w = frame.shape[:2]
        self.update_frame_shape(h, w)

        # Edge pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self._canny_lo, self._canny_hi)

        # ROI
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self._roi_vertices, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        # Hough
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=self._rho,
            theta=self._theta,
            threshold=self._hough_thr,
            minLineLength=self._min_line,
            maxLineGap=self._max_gap,
        )

        left_segs, right_segs = self._classify_lines(lines)
        left_lane = self._average_lane(left_segs, "left")
        right_lane = self._average_lane(right_segs, "right")

        lane_center, vehicle_offset = self._calculate_offset(left_lane, right_lane)
        departure = abs(vehicle_offset) > self._departure_thr
        departure_risk = min(1.0, abs(vehicle_offset) / (self._departure_thr * 2))

        return LaneResult(
            left_lane=left_lane,
            right_lane=right_lane,
            lane_center=lane_center,
            vehicle_offset=vehicle_offset,
            departure_warning=departure,
            departure_risk=departure_risk,
        )

    def _classify_lines(
        self, lines: Optional[np.ndarray]
    ) -> Tuple[List[Tuple], List[Tuple]]:
        left, right = [], []
        if lines is None:
            return left, right
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < 0.3:  # near-horizontal noise
                continue
            if slope < 0:
                left.append((x1, y1, x2, y2))
            else:
                right.append((x1, y1, x2, y2))
        return left, right

    def _average_lane(
        self, segments: List[Tuple], side: str
    ) -> Optional[Tuple[int, int, int, int]]:
        if not segments:
            return None

        xs, ys = [], []
        for x1, y1, x2, y2 in segments:
            xs += [x1, x2]
            ys += [y1, y2]

        try:
            poly = np.polyfit(ys, xs, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        y_bot = self._h
        y_top = int(self._h * self._roi_top)
        x_bot = int(np.polyval(poly, y_bot))
        x_top = int(np.polyval(poly, y_top))

        history = self._left_history if side == "left" else self._right_history
        history.append((x_bot, y_bot, x_top, y_top))

        avg = np.mean(list(history), axis=0).astype(int)
        return (int(avg[0]), int(avg[1]), int(avg[2]), int(avg[3]))

    def _calculate_offset(
        self,
        left: Optional[Tuple],
        right: Optional[Tuple],
    ) -> Tuple[int, float]:
        frame_center = self._w // 2
        approx_lane_width = int(self._w * 0.45)

        if left and right:
            lane_center = (left[0] + right[0]) // 2
        elif left:
            lane_center = left[0] + approx_lane_width // 2
        elif right:
            lane_center = right[0] - approx_lane_width // 2
        else:
            return frame_center, 0.0

        offset = (frame_center - lane_center) / (approx_lane_width / 2)
        return lane_center, float(np.clip(offset, -1.0, 1.0))

    def draw_lanes(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """Draw detected lane lines and centre indicator on a copy of frame."""
        out = frame.copy()
        h = out.shape[0]

        lane_colour = (0, 0, 255) if result.departure_warning else (0, 255, 0)

        if result.left_lane:
            x1, y1, x2, y2 = result.left_lane
            cv2.line(out, (x1, y1), (x2, y2), lane_colour, 4)

        if result.right_lane:
            x1, y1, x2, y2 = result.right_lane
            cv2.line(out, (x1, y1), (x2, y2), lane_colour, 4)

        # Lane centre line
        cv2.line(out, (result.lane_center, int(h * 0.6)), (result.lane_center, h), (255, 255, 0), 2)

        # Frame centre
        frame_cx = out.shape[1] // 2
        cv2.line(out, (frame_cx, int(h * 0.6)), (frame_cx, h), (255, 0, 0), 1)

        if result.departure_warning:
            direction = "LEFT" if result.vehicle_offset > 0 else "RIGHT"
            cv2.putText(
                out,
                f"LANE DEPARTURE {direction}",
                (out.shape[1] // 2 - 160, int(h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return out
