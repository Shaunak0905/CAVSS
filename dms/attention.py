"""
dms/attention.py
Head pose estimation and distraction detection.
Uses solvePnP with 6 facial landmarks to get pitch, yaw, roll in degrees.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 3-D reference face model points (generic, in mm)
# Order: nose tip, left eye outer, right eye outer,
#        left mouth corner, right mouth corner, chin
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),            # Nose tip
        (-30.0, -125.0, -30.0),     # Left eye outer corner
        (30.0, -125.0, -30.0),      # Right eye outer corner
        (-60.0, -70.0, -60.0),      # Left mouth corner
        (60.0, -70.0, -60.0),       # Right mouth corner
        (0.0, -330.0, -65.0),       # Chin
    ],
    dtype=np.float64,
)


@dataclass
class AttentionState:
    """Head pose and distraction metrics returned each frame."""
    pitch: float = 0.0           # Degrees: positive = head down (nodding)
    yaw: float = 0.0             # Degrees: positive = head right
    roll: float = 0.0            # Degrees: tilt
    is_distracted: bool = False
    distraction_duration_s: float = 0.0
    attention_score: float = 0.0  # 0–1: 1 = fully distracted
    processing_time_ms: float = 0.0


class AttentionDetector:
    """
    Estimates head pose and tracks distraction duration.

    Args:
        config: Dict from config.yaml['dms']['attention'].
        frame_shape: (height, width) of the webcam feed — used to build
                     the camera matrix. Must be set before first call to
                     update(), or passed explicitly.
    """

    def __init__(self, config: dict, frame_shape: Tuple[int, int] = (480, 640)) -> None:
        a = config
        self.pitch_threshold: float = a.get("pitch_threshold", 20.0)
        self.yaw_threshold: float = a.get("yaw_threshold", 30.0)
        self.roll_threshold: float = a.get("roll_threshold", 25.0)
        self.distraction_duration: float = a.get("distraction_duration_seconds", 2.0)
        self._alpha: float = a.get("smoothing_factor", 0.3)

        # Build camera matrix from frame dimensions
        h, w = frame_shape
        self._frame_shape = (h, w)
        self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(h, w)

        # State
        self._distraction_start: Optional[float] = None
        self._smoothed_pitch = 0.0
        self._smoothed_yaw = 0.0
        self._smoothed_roll = 0.0

        logger.info(
            f"AttentionDetector ready (pitch={self.pitch_threshold}°, "
            f"yaw={self.yaw_threshold}°, duration={self.distraction_duration}s)"
        )

    def _build_camera_matrix(
        self, h: int, w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        focal = float(w)
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))
        return camera_matrix, dist_coeffs

    def update_frame_shape(self, h: int, w: int) -> None:
        """Rebuild camera matrix if frame dimensions change."""
        if (h, w) != self._frame_shape:
            self._frame_shape = (h, w)
            self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(h, w)

    def update(self, head_pose_points: np.ndarray) -> AttentionState:
        """
        Estimate head pose from 6 landmark points.

        Args:
            head_pose_points: (6, 2) float64 array of pixel coordinates
                              from FaceMeshProcessor.get_head_pose_landmarks().

        Returns:
            AttentionState with pitch/yaw/roll and distraction metrics.
        """
        t0 = time.perf_counter()

        pitch, yaw, roll = self._solve_pose(head_pose_points)

        # Smooth
        self._smoothed_pitch = self._alpha * pitch + (1 - self._alpha) * self._smoothed_pitch
        self._smoothed_yaw = self._alpha * yaw + (1 - self._alpha) * self._smoothed_yaw
        self._smoothed_roll = self._alpha * roll + (1 - self._alpha) * self._smoothed_roll

        sp, sy, sr = self._smoothed_pitch, self._smoothed_yaw, self._smoothed_roll

        # Distraction if any angle exceeds threshold
        over_pitch = abs(sp) > self.pitch_threshold
        over_yaw = abs(sy) > self.yaw_threshold
        over_roll = abs(sr) > self.roll_threshold
        posture_distracted = over_pitch or over_yaw or over_roll

        # Duration tracking
        now = time.time()
        if posture_distracted:
            if self._distraction_start is None:
                self._distraction_start = now
            duration = now - self._distraction_start
        else:
            self._distraction_start = None
            duration = 0.0

        is_distracted = duration >= self.distraction_duration

        # Attention score (0–1): how much over the worst threshold?
        max_deviation = max(
            abs(sp) / self.pitch_threshold,
            abs(sy) / self.yaw_threshold,
            abs(sr) / self.roll_threshold,
        )
        attention_score = min(1.0, max(0.0, (max_deviation - 1.0) * 0.5)) if posture_distracted else 0.0
        # Also scale by duration
        if duration > 0:
            duration_scale = min(1.0, duration / (self.distraction_duration * 2))
            attention_score = min(1.0, attention_score + duration_scale * 0.3)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AttentionState(
            pitch=float(sp),
            yaw=float(sy),
            roll=float(sr),
            is_distracted=is_distracted,
            distraction_duration_s=duration,
            attention_score=attention_score,
            processing_time_ms=elapsed_ms,
        )

    def _solve_pose(
        self, image_points: np.ndarray
    ) -> Tuple[float, float, float]:
        """Run cv2.solvePnP and return (pitch, yaw, roll) in degrees."""
        success, rotation_vec, translation_vec = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])
        return pitch, yaw, roll

    def reset(self) -> None:
        """Reset distraction tracking state."""
        self._distraction_start = None
        self._smoothed_pitch = 0.0
        self._smoothed_yaw = 0.0
        self._smoothed_roll = 0.0
