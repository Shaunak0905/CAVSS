"""
interface/dashboard.py
Real-time demo dashboard — tiles DMS (webcam) and ADAS (road) side by side
with the CRE risk panel displayed between them.

Layout (1280 × 720):
  ┌─────────────────┬──────────────────┐
  │  DMS feed       │   ADAS feed      │
  │  640 × 480      │   640 × 480      │
  │  (webcam face)  │   (road + YOLO)  │
  ├─────────────────┴──────────────────┤
  │  Risk panel  (1280 × 240)          │
  └────────────────────────────────────┘
"""

import logging
import time
from typing import Dict, Optional
import numpy as np
import cv2

from context_engine.risk_calculator import RiskOutput, ContextState
from alerts.visual_alert import LEVEL_COLOURS, LEVEL_LABELS

logger = logging.getLogger(__name__)

# Panel heights
_FEED_H = 480
_PANEL_H = 240
_FEED_W = 640
_TOTAL_W = 1280
_TOTAL_H = _FEED_H + _PANEL_H   # 720


class Dashboard:
    """
    Composites the DMS feed, ADAS feed, and CRE risk panel into a single
    1280×720 frame for display and optional recording.

    Args:
        window_name: OpenCV window title.
        record: If True, write frames to output_path.
        output_path: Video file path for recording.
        fps: Recording FPS.
    """

    def __init__(
        self,
        window_name: str = "CAVSS — Context-Aware Vehicle Safety System",
        record: bool = False,
        output_path: str = "output/recordings/demo.mp4",
        fps: int = 15,
    ) -> None:
        self.window_name = window_name
        self._record = record
        self._writer: Optional[cv2.VideoWriter] = None
        self._fps = fps
        self._frame_count = 0

        if record:
            self._init_writer(output_path)

    def _init_writer(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, self._fps, (_TOTAL_W, _TOTAL_H))
        if self._writer.isOpened():
            logger.info(f"Recording to {path}")
        else:
            logger.warning("VideoWriter failed to open")
            self._writer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        dms_frame: Optional[np.ndarray],
        adas_frame: Optional[np.ndarray],
        risk_output: RiskOutput,
        context: ContextState,
        system_fps: float = 0.0,
        show_debug: bool = True,
    ) -> np.ndarray:
        """
        Compose and return the 1280×720 dashboard frame.

        Args:
            dms_frame: Annotated webcam frame (any size — resized to 640×480).
            adas_frame: Annotated road frame (any size — resized to 640×480).
            risk_output: Latest CRE output.
            context: Current driving context.
            system_fps: Overall loop FPS for display.
            show_debug: If True, show component risk bars.

        Returns:
            1280×720 BGR numpy array.
        """
        canvas = np.zeros((_TOTAL_H, _TOTAL_W, 3), dtype=np.uint8)

        # -- DMS feed (left half) --
        dms_tile = self._prepare_tile(dms_frame, _FEED_W, _FEED_H, "DMS — Driver Monitor")
        canvas[0:_FEED_H, 0:_FEED_W] = dms_tile

        # -- ADAS feed (right half) --
        adas_tile = self._prepare_tile(adas_frame, _FEED_W, _FEED_H, "ADAS — Road Monitor")
        canvas[0:_FEED_H, _FEED_W:_TOTAL_W] = adas_tile

        # Divider line
        cv2.line(canvas, (_FEED_W, 0), (_FEED_W, _FEED_H), (60, 60, 60), 2)

        # -- Risk panel (bottom strip) --
        panel = self._build_risk_panel(risk_output, context, system_fps, show_debug)
        canvas[_FEED_H:_TOTAL_H, 0:_TOTAL_W] = panel

        self._frame_count += 1
        if self._writer:
            self._writer.write(canvas)

        return canvas

    def show(self, canvas: np.ndarray) -> int:
        """
        Display canvas in an OpenCV window.

        Returns:
            Key pressed (0xFF masked) or -1 if no key.
        """
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(1) & 0xFF

    def screenshot(self, canvas: np.ndarray, path: str) -> None:
        """Save current frame to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, canvas)
        logger.info(f"Screenshot saved: {path}")

    def stop(self) -> None:
        """Release VideoWriter and close windows."""
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_tile(
        self,
        frame: Optional[np.ndarray],
        w: int,
        h: int,
        label: str,
    ) -> np.ndarray:
        """Resize frame to tile dimensions or return placeholder."""
        if frame is None or frame.size == 0:
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, f"Waiting: {label}",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1,
            )
            return placeholder
        fh, fw = frame.shape[:2]
        if fw != w or fh != h:
            frame = cv2.resize(frame, (w, h))
        return frame

    def _build_risk_panel(
        self,
        risk: RiskOutput,
        context: ContextState,
        fps: float,
        show_debug: bool,
    ) -> np.ndarray:
        """Build the bottom risk panel as a 1280×240 BGR array."""
        panel = np.zeros((_PANEL_H, _TOTAL_W, 3), dtype=np.uint8)
        colour = LEVEL_COLOURS.get(risk.alert_level, (200, 200, 200))
        label = LEVEL_LABELS.get(risk.alert_level, risk.alert_level.upper())
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Panel border
        cv2.rectangle(panel, (0, 0), (_TOTAL_W - 1, _PANEL_H - 1), colour, 2)

        # Risk score gauge (large text, centre-left)
        score_text = f"{risk.final_risk_score:.2f}"
        cv2.putText(panel, score_text, (20, 100), font, 3.5, colour, 5, cv2.LINE_AA)

        # Alert level label
        cv2.putText(panel, label, (20, 145), font, 1.1, colour, 3, cv2.LINE_AA)

        # Score bar
        bar_x, bar_y = 180, 30
        bar_w, bar_h = 460, 28
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled = int(bar_w * risk.final_risk_score)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), colour, -1)
        # Threshold markers
        for thr, thr_label in [(0.3, "Y"), (0.5, "O"), (0.7, "R"), (0.85, "C")]:
            mx = bar_x + int(bar_w * thr)
            cv2.line(panel, (mx, bar_y), (mx, bar_y + bar_h), (200, 200, 200), 1)
            cv2.putText(panel, thr_label, (mx - 5, bar_y - 3), font, 0.4, (160, 160, 160), 1)

        # Alert message
        if risk.alert_message:
            cv2.putText(panel, risk.alert_message, (180, 90), font, 0.65, colour, 1, cv2.LINE_AA)

        # Context info (right column)
        ctx_x = 680
        ctx_lines = [
            f"Zone:    {context.zone_type}",
            f"Speed:   {context.speed_kmh:.0f} km/h",
            f"Time:    {context.time_of_day.strftime('%H:%M')}",
            f"Visibility: {context.visibility}",
            f"FPS:     {fps:.1f}",
        ]
        for i, line in enumerate(ctx_lines):
            cv2.putText(panel, line, (ctx_x, 45 + i * 38), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        # Context multipliers
        mult_x = 920
        mults = risk.active_multipliers
        mult_lines = [
            f"Time  x{mults.get('time', 1.0):.1f}",
            f"Zone  x{mults.get('zone', 1.0):.1f}",
            f"Vis   x{mults.get('visibility', 1.0):.1f}",
            f"Combined x{mults.get('combined', 1.0):.2f}",
        ]
        for i, line in enumerate(mult_lines):
            cv2.putText(panel, line, (mult_x, 45 + i * 38), font, 0.65, (140, 200, 140), 1, cv2.LINE_AA)

        # Component risk bars (debug)
        if show_debug:
            self._draw_component_bars(panel, risk.component_risks)

        # Team info
        cv2.putText(panel, "CAVSS v1.0 | MIT-WPU | Team 25266P4-73",
                    (680, 215), font, 0.45, (80, 80, 80), 1)

        return panel

    def _draw_component_bars(
        self,
        panel: np.ndarray,
        components: Dict[str, float],
    ) -> None:
        """Draw mini bar chart for each risk component."""
        bar_names = [
            ("drowsiness",       (0, 0, 230)),
            ("attention",        (0, 130, 255)),
            ("speed",            (0, 200, 255)),
            ("lane_departure",   (0, 255, 0)),
            ("forward_collision",(0, 100, 255)),
            ("visibility",       (150, 150, 150)),
        ]
        start_x = 180
        start_y = 155
        bw = 60
        gap = 12
        max_bar_h = 50
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, (name, colour) in enumerate(bar_names):
            val = components.get(name, 0.0)
            bh = int(max_bar_h * min(1.0, val))
            x = start_x + i * (bw + gap)
            # Background
            cv2.rectangle(panel, (x, start_y), (x + bw, start_y + max_bar_h), (40, 40, 40), -1)
            # Bar
            if bh > 0:
                cv2.rectangle(panel, (x, start_y + max_bar_h - bh), (x + bw, start_y + max_bar_h), colour, -1)
            # Value label
            cv2.putText(panel, f"{val:.2f}", (x + 2, start_y + max_bar_h + 14),
                        font, 0.38, (180, 180, 180), 1)
            # Name label (abbreviated)
            abbrev = name[:4]
            cv2.putText(panel, abbrev, (x + 2, start_y - 3),
                        font, 0.38, (140, 140, 140), 1)
