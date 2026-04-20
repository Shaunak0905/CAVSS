"""
dms/drowsiness.py
Drowsiness detection via EAR (Eye Aspect Ratio) and PERCLOS.
Also detects yawning via Mouth Aspect Ratio (MAR).
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Default thresholds (overridden from config in DrowsinessDetector.__init__)
_EAR_THRESHOLD = 0.25
_PERCLOS_THRESHOLD = 0.40
_MAR_THRESHOLD = 0.60


def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Eye Aspect Ratio from 6 eye landmarks.

    Landmark order: [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: List of 6 (x, y) pixel coordinate tuples.

    Returns:
        EAR value (typically 0.15–0.35 for open eyes, <0.25 = closed).
    """
    p = [np.array(pt) for pt in eye_landmarks]
    v1 = float(np.linalg.norm(p[1] - p[5]))
    v2 = float(np.linalg.norm(p[2] - p[4]))
    h = float(np.linalg.norm(p[0] - p[3]))
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def calculate_mar(mouth_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Mouth Aspect Ratio for yawn detection.

    Uses a simplified 4-point MAR: top, bottom, left, right mouth corners.

    Args:
        mouth_landmarks: List of (x, y) tuples (at least 4 points).

    Returns:
        MAR value — higher value = wider open mouth.
    """
    if len(mouth_landmarks) < 4:
        return 0.0
    pts = [np.array(p) for p in mouth_landmarks]
    # Vertical: point 2 (top) and 3 (bottom)
    vertical = float(np.linalg.norm(pts[2] - pts[3]))
    # Horizontal: point 0 (left corner) and 1 (right corner)
    horizontal = float(np.linalg.norm(pts[0] - pts[1]))
    return vertical / horizontal if horizontal > 1e-6 else 0.0


@dataclass
class DrowsinessState:
    """Current drowsiness state returned each frame."""
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    perclos: float = 0.0
    mar: float = 0.0
    is_eyes_closed: bool = False
    is_yawning: bool = False
    consecutive_closed_frames: int = 0
    yawn_count: int = 0
    drowsiness_score: float = 0.0  # Normalized 0–1 for CRE
    processing_time_ms: float = 0.0


class PERCLOSCalculator:
    """
    Rolling-window PERCLOS calculator.

    PERCLOS = fraction of frames in a time window where eyes were closed.
    Industry threshold: PERCLOS > 0.40 over 60s = severe drowsiness.

    Args:
        window_seconds: Duration of the rolling window (default 60s).
        fps: Expected frame rate to pre-size the deque.
        ear_threshold: EAR below which eyes are considered closed.
    """

    def __init__(
        self,
        window_seconds: int = 60,
        fps: int = 30,
        ear_threshold: float = _EAR_THRESHOLD,
    ) -> None:
        self.window_size = window_seconds * fps
        self.ear_threshold = ear_threshold
        self._eye_states: deque = deque(maxlen=self.window_size)

    def update(self, ear: float) -> float:
        """
        Add new EAR reading and return current PERCLOS.

        Returns:
            PERCLOS as a fraction (0.0–1.0). Returns 0 until at least
            30 samples are available.
        """
        self._eye_states.append(1 if ear < self.ear_threshold else 0)
        if len(self._eye_states) < 30:
            return 0.0
        return float(sum(self._eye_states)) / len(self._eye_states)


class DrowsinessDetector:
    """
    Combines EAR, PERCLOS, and yawn detection into a single drowsiness score.

    Args:
        config: Dictionary from config.yaml['dms']['drowsiness'].
        fps: Webcam FPS (used to size PERCLOS window).
    """

    def __init__(self, config: dict, fps: int = 30) -> None:
        d = config
        self.ear_threshold: float = d.get("ear_threshold", _EAR_THRESHOLD)
        self.ear_consec_frames: int = d.get("ear_consecutive_frames", 3)
        self.mar_threshold: float = d.get("mar_threshold", _MAR_THRESHOLD)
        self.yawn_consec_frames: int = d.get("yawn_consecutive_frames", 5)
        perclos_window: int = d.get("perclos_window_seconds", 60)
        perclos_threshold: float = d.get("perclos_threshold", _PERCLOS_THRESHOLD)

        self._perclos = PERCLOSCalculator(
            window_seconds=perclos_window,
            fps=fps,
            ear_threshold=self.ear_threshold,
        )
        self.perclos_threshold = perclos_threshold
        self._consec_closed = 0
        self._consec_yawn = 0
        self._yawn_count = 0
        self._was_yawning = False

        # Exponential smoothing to reduce jitter
        self._smoothed_ear = 0.3
        self._alpha = 0.3

        logger.info(
            f"DrowsinessDetector ready (EAR_thr={self.ear_threshold}, "
            f"PERCLOS_thr={perclos_threshold}, window={perclos_window}s)"
        )

    def update(
        self,
        left_eye: List[Tuple[float, float]],
        right_eye: List[Tuple[float, float]],
        mouth: List[Tuple[float, float]],
    ) -> DrowsinessState:
        """
        Process one frame of landmark data.

        Args:
            left_eye: 6 (x, y) landmarks from FaceMeshProcessor.
            right_eye: 6 (x, y) landmarks from FaceMeshProcessor.
            mouth: 8 (x, y) landmarks for yawn detection.

        Returns:
            DrowsinessState with all metrics and the combined score.
        """
        t0 = time.perf_counter()

        # --- EAR ---
        ear_l = calculate_ear(left_eye)
        ear_r = calculate_ear(right_eye)
        ear_avg = (ear_l + ear_r) / 2.0

        # Smooth EAR
        self._smoothed_ear = self._alpha * ear_avg + (1 - self._alpha) * self._smoothed_ear
        smoothed = self._smoothed_ear

        # Closed-eye counter
        if smoothed < self.ear_threshold:
            self._consec_closed += 1
        else:
            self._consec_closed = 0

        is_closed = self._consec_closed >= self.ear_consec_frames

        # --- PERCLOS ---
        perclos = self._perclos.update(smoothed)

        # --- MAR / Yawn ---
        mar = calculate_mar(mouth)
        if mar > self.mar_threshold:
            self._consec_yawn += 1
        else:
            self._consec_yawn = 0

        is_yawning = self._consec_yawn >= self.yawn_consec_frames
        if is_yawning and not self._was_yawning:
            self._yawn_count += 1
        self._was_yawning = is_yawning

        # --- Drowsiness Score (0–1) ---
        # EAR component: inversely proportional to EAR
        ear_score = max(0.0, 1.0 - (smoothed / self.ear_threshold)) if is_closed else 0.0

        # PERCLOS component: how far above threshold
        perclos_score = min(1.0, perclos / self.perclos_threshold)

        # Yawn adds a small bonus
        yawn_score = 0.15 if is_yawning else 0.0

        # Combine: PERCLOS is the stronger long-term signal
        drowsiness_score = min(1.0, ear_score * 0.4 + perclos_score * 0.5 + yawn_score)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DrowsinessState(
            ear_left=ear_l,
            ear_right=ear_r,
            ear_avg=ear_avg,
            perclos=perclos,
            mar=mar,
            is_eyes_closed=is_closed,
            is_yawning=is_yawning,
            consecutive_closed_frames=self._consec_closed,
            yawn_count=self._yawn_count,
            drowsiness_score=drowsiness_score,
            processing_time_ms=elapsed_ms,
        )

    def reset(self) -> None:
        """Reset counters (e.g., when face is lost)."""
        self._consec_closed = 0
        self._consec_yawn = 0
        self._smoothed_ear = 0.3
