"""
alerts/alert_manager.py
Orchestrates AudioAlert and VisualAlert — single entry point for all alerts.
"""

import logging
from typing import Optional, Dict
import numpy as np

from .audio_alert import AudioAlert
from .visual_alert import VisualAlert
from context_engine.risk_calculator import RiskOutput

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Single entry point that fires audio + visual alerts from a RiskOutput.

    Args:
        config: Full config.yaml dict (passes sub-dicts to sub-systems).
    """

    def __init__(self, config: dict) -> None:
        alerts_cfg = config.get("alerts", {})
        self._audio = AudioAlert(alerts_cfg)
        self._visual = VisualAlert(alerts_cfg.get("visual", {}))
        logger.info("AlertManager initialised")

    def process(self, risk_output: RiskOutput) -> None:
        """
        Fire audio alerts based on risk_output.

        Args:
            risk_output: From ContextRiskEngine.calculate_risk().
        """
        self._audio.alert(risk_output.alert_level, risk_output.voice_alert)

    def annotate_dms(
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
        """Annotate webcam frame with DMS metrics."""
        return self._visual.draw_dms_overlay(
            frame, ear, perclos, pitch, yaw, drowsy, distracted, yawning, face_detected
        )

    def annotate_adas(
        self,
        frame: np.ndarray,
        fps: float,
        vehicle_count: int,
        pedestrian_count: int,
        lane_offset: float,
        ttc: Optional[float],
        visibility: str,
    ) -> np.ndarray:
        """Annotate road frame with ADAS metrics."""
        return self._visual.draw_adas_overlay(
            frame, fps, vehicle_count, pedestrian_count, lane_offset, ttc, visibility
        )

    def annotate_risk(
        self,
        frame: np.ndarray,
        risk_output: RiskOutput,
    ) -> np.ndarray:
        """Draw the risk overlay (score bar, level badge, message) on frame."""
        return self._visual.draw_risk_overlay(
            frame,
            risk_score=risk_output.final_risk_score,
            alert_level=risk_output.alert_level,
            alert_message=risk_output.alert_message,
            component_risks=risk_output.component_risks,
            multipliers=risk_output.active_multipliers,
        )

    def stop(self) -> None:
        """Stop all audio."""
        self._audio.stop()
