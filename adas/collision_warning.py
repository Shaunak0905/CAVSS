"""
adas/collision_warning.py
Time-to-Collision (TTC) estimation based on bounding box size changes.
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .object_detection import Detection

logger = logging.getLogger(__name__)


@dataclass
class CollisionRisk:
    """Collision risk assessment output."""
    ttc: Optional[float]           # seconds (None = no threat)
    risk_level: str                # "safe", "warning", "critical"
    threat_object: Optional[Detection]
    forward_risk_score: float      # 0–1 for CRE


class CollisionWarning:
    """
    Estimates Time-to-Collision from bounding box height growth.

    Distance is approximated using the pinhole camera model:
        distance = (ref_height_px × ref_distance_m) / bbox_height_px

    TTC is then:
        TTC = current_distance / approach_velocity

    Args:
        config: Dict from config.yaml['adas']['collision'].
        frame_fps: Assumed FPS of the ADAS feed for velocity calculation.
        frame_width: Frame width in pixels (used for "in-path" check).
    """

    def __init__(
        self,
        config: dict,
        frame_fps: float = 15.0,
        frame_width: int = 1280,
    ) -> None:
        self._ttc_warning: float = config.get("ttc_warning", 3.0)
        self._ttc_critical: float = config.get("ttc_critical", 1.5)

        # Camera calibration approx (calibrate for real deployment)
        self._ref_height_m: float = config.get("reference_height_meters", 1.5)
        self._focal_px: float = config.get("focal_length_pixels", 800)

        self._frame_time: float = 1.0 / max(frame_fps, 1.0)
        self._frame_width = frame_width

        # Previous distances keyed by a tracking ID string
        self._prev_distances: Dict[str, float] = {}
        self._prev_time: float = time.time()

        logger.info(
            f"CollisionWarning ready (warn={self._ttc_warning}s, "
            f"crit={self._ttc_critical}s)"
        )

    def _estimate_distance(self, bbox: Tuple[int, int, int, int]) -> float:
        """Approximate distance to object from bounding box height."""
        _, y1, _, y2 = bbox
        box_h = max(y2 - y1, 1)
        # distance = focal_length * ref_height_m / box_height_px
        # Simplified: use reference values
        distance = (self._focal_px * self._ref_height_m) / box_h
        return float(distance)

    def assess_risk(self, detections: List[Detection]) -> CollisionRisk:
        """
        Evaluate TTC for all detections and return the worst-case risk.

        Only objects in the central 40% of the frame (in our path) are
        considered for collision risk.

        Args:
            detections: From ObjectDetector.detect().

        Returns:
            CollisionRisk with TTC and risk_level.
        """
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt <= 0:
            dt = self._frame_time

        min_ttc = float("inf")
        worst_det: Optional[Detection] = None
        new_distances: Dict[str, float] = {}

        for i, det in enumerate(detections):
            cx = det.center[0]
            # Only consider objects roughly in our path
            if not (0.25 * self._frame_width < cx < 0.75 * self._frame_width):
                continue

            det_id = f"{det.class_name}_{i}"
            current_dist = self._estimate_distance(det.bbox)
            new_distances[det_id] = current_dist

            if det_id in self._prev_distances:
                prev_dist = self._prev_distances[det_id]
                approach_vel = (prev_dist - current_dist) / dt
                # Only care if closing at > 0.5 m/s
                if approach_vel > 0.5 and current_dist > 0:
                    ttc = current_dist / approach_vel
                    det.ttc = ttc
                    det.distance_estimate = current_dist
                    if ttc < min_ttc:
                        min_ttc = ttc
                        worst_det = det

        self._prev_distances = new_distances

        # Clip stale entries
        if len(self._prev_distances) > 50:
            keys = list(self._prev_distances.keys())
            for k in keys[:-50]:
                del self._prev_distances[k]

        risk_level = self._get_risk_level(min_ttc)
        forward_risk = self._ttc_to_score(min_ttc)

        return CollisionRisk(
            ttc=min_ttc if min_ttc < float("inf") else None,
            risk_level=risk_level,
            threat_object=worst_det,
            forward_risk_score=forward_risk,
        )

    def _get_risk_level(self, ttc: float) -> str:
        if ttc < self._ttc_critical:
            return "critical"
        elif ttc < self._ttc_warning:
            return "warning"
        return "safe"

    def _ttc_to_score(self, ttc: float) -> float:
        """Convert TTC to a 0–1 risk score for the CRE."""
        if ttc == float("inf"):
            return 0.0
        if ttc <= self._ttc_critical:
            return 1.0
        if ttc >= self._ttc_warning * 2:
            return 0.0
        # Linear interpolation between warning and critical
        return 1.0 - (ttc - self._ttc_critical) / (self._ttc_warning * 2 - self._ttc_critical)
