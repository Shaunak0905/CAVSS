"""
alerts/visual_alert.py
Visual alert overlay — draws colour-coded risk panel onto OpenCV frames.
"""

import time
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Alert level → BGR colour (OpenCV is BGR)
LEVEL_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "green":    (0, 200, 0),
    "yellow":   (0, 220, 220),
    "orange":   (0, 140, 255),
    "red":      (0, 0, 220),
    "critical": (0, 0, 255),
}

LEVEL_LABELS: Dict[str, str] = {
    "green":    "SAFE",
    "yellow":   "CAUTION",
    "orange":   "WARNING",
    "red":      "DANGER",
    "critical": "CRITICAL",
}


class VisualAlert:
    """
    Renders a real-time risk overlay onto dashboard frames.

    Draws:
    - Coloured risk score bar (bottom of dashboard)
    - Alert level badge
    - Alert message text
    - Flashing border on critical

    Args:
        config: Dict from config.yaml['alerts']['visual'].
        flash_interval_ms: Milliseconds between critical flashes.
    """

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled", True)
        self._flash_interval_ms: float = config.get("flash_interval_ms", 200)
        self._flash_state = False
        self._last_flash_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_risk_overlay(
        self,
        frame: np.ndarray,
        risk_score: float,
        alert_level: str,
        alert_message: Optional[str] = None,
        component_risks: Optional[Dict[str, float]] = None,
        multipliers: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Draw risk overlay on a copy of the frame.

        Args:
            frame: BGR dashboard frame.
            risk_score: 0–1 float from CRE.
            alert_level: "green" | "yellow" | "orange" | "red" | "critical".
            alert_message: Optional text to display.
            component_risks: Breakdown dict for debug overlay.
            multipliers: Context multipliers dict for debug overlay.

        Returns:
            Annotated copy of the frame.
        """
        if not self._enabled:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        colour = LEVEL_COLOURS.get(alert_level, (200, 200, 200))

        # Flashing border on critical
        if alert_level == "critical":
            self._update_flash()
            if self._flash_state:
                cv2.rectangle(out, (0, 0), (w - 1, h - 1), colour, 8)
        else:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), colour, 3)

        # Risk score bar (bottom strip)
        bar_h = 18
        bar_y = h - bar_h
        filled_w = int(w * risk_score)
        cv2.rectangle(out, (0, bar_y), (w, h), (30, 30, 30), -1)
        cv2.rectangle(out, (0, bar_y), (filled_w, h), colour, -1)

        # Alert badge (top-right)
        label = LEVEL_LABELS.get(alert_level, alert_level.upper())
        badge_text = f"  {label}  {risk_score:.2f}  "
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.7, 2)
        badge_x = w - tw - 10
        badge_y = 10
        cv2.rectangle(out, (badge_x - 4, badge_y), (w - 6, badge_y + th + 8), colour, -1)
        cv2.putText(
            out, badge_text,
            (badge_x, badge_y + th + 2),
            font, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )

        # Alert message text (top-left, below DMS info)
        if alert_message and alert_level != "green":
            msg_y = 55
            cv2.rectangle(out, (0, msg_y - 22), (min(w, len(alert_message) * 11 + 20), msg_y + 8), (0, 0, 0), -1)
            cv2.putText(
                out, alert_message,
                (10, msg_y),
                font, 0.62, colour, 2, cv2.LINE_AA,
            )

        return out

    def draw_dms_overlay(
        self,
        frame: np.ndarray,
        ear: float,
        perclos: float,
        pitch: float,
        yaw: float,
        drowsy: bool,
        distracted: bool,
        yawning: bool,
        face_detected: bool,
    ) -> np.ndarray:
        """Draw DMS metrics in the top-left corner of a webcam frame."""
        out = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"EAR: {ear:.3f}",
            f"PERCLOS: {perclos:.1%}",
            f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}",
        ]
        statuses = []
        if not face_detected:
            statuses.append(("NO FACE", (0, 128, 255)))
        else:
            if drowsy:
                statuses.append(("DROWSY", (0, 0, 255)))
            if distracted:
                statuses.append(("DISTRACTED", (0, 140, 255)))
            if yawning:
                statuses.append(("YAWNING", (0, 200, 255)))

        # Background panel
        panel_h = len(lines) * 22 + len(statuses) * 28 + 10
        cv2.rectangle(out, (0, 0), (220, panel_h), (0, 0, 0), -1)

        y = 18
        for line in lines:
            cv2.putText(out, line, (6, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22
        for text, colour in statuses:
            cv2.putText(out, text, (6, y), font, 0.75, colour, 2, cv2.LINE_AA)
            y += 28

        return out

    def draw_adas_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        vehicle_count: int,
        pedestrian_count: int,
        lane_offset: float,
        ttc: Optional[float],
        visibility: str,
    ) -> np.ndarray:
        """Draw ADAS metrics in the top-left corner of the road frame."""
        out = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"FPS: {fps:.1f}",
            f"Vehicles: {vehicle_count}  Peds: {pedestrian_count}",
            f"Lane offset: {lane_offset:+.2f}",
            f"Visibility: {visibility}",
        ]
        if ttc is not None:
            lines.append(f"TTC: {ttc:.1f}s")

        panel_h = len(lines) * 22 + 10
        cv2.rectangle(out, (0, 0), (240, panel_h), (0, 0, 0), -1)

        y = 18
        for line in lines:
            cv2.putText(out, line, (6, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_flash(self) -> None:
        now_ms = time.time() * 1000
        if now_ms - self._last_flash_time >= self._flash_interval_ms:
            self._flash_state = not self._flash_state
            self._last_flash_time = now_ms
