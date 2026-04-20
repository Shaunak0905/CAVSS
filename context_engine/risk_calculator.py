"""
context_engine/risk_calculator.py
The Context Risk Engine (CRE) — CAVSS's core innovation.

Formula: Final_Risk = Σ(Wi × Ci × Si)
  Wi = base weight for parameter i
  Ci = combined context multiplier (time × zone × visibility, capped at 2.5)
  Si = sensor reading, normalized 0–1
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple
import yaml

from .time_context import TimeContext
from .zone_manager import ZoneManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContextState:
    """Current driving context — set by ScenarioSimulator or real sensors."""
    time_of_day: datetime
    zone_type: str   # "urban", "highway", "blackspot", "school_zone", "rural", "residential"
    visibility: str  # "clear", "overcast", "rain", "fog", "night", "heavy_rain", "dusk_dawn"
    speed_kmh: float
    location: Optional[Dict] = None  # {"lat": ..., "lon": ...}


@dataclass
class SensorReadings:
    """Normalised sensor readings (0–1) from DMS and ADAS modules."""
    drowsiness: float = 0.0
    attention: float = 0.0
    lane_departure: float = 0.0
    forward_collision: float = 0.0


@dataclass
class RiskOutput:
    """Full output from the CRE for dashboard and alert system."""
    final_risk_score: float
    alert_level: str   # "green", "yellow", "orange", "red", "critical"

    component_risks: Dict[str, float] = field(default_factory=dict)
    active_multipliers: Dict[str, float] = field(default_factory=dict)
    dominant_risk: str = "none"

    alert_message: Optional[str] = None
    voice_alert: Optional[str] = None


# ---------------------------------------------------------------------------
# Demo scenario simulator
# ---------------------------------------------------------------------------

class ScenarioSimulator:
    """
    Pre-defined demo scenarios switchable via keyboard.

    Provides a ContextState for each scenario so the demo can showcase
    the CRE's context-sensitivity.
    """

    SCENARIOS: Dict[str, Dict] = {
        "normal": {
            "zone": "urban",
            "visibility": "clear",
            "speed": 40,
            "hour": 14,
            "description": "Normal daytime city driving",
        },
        "highway_night": {
            "zone": "highway",
            "visibility": "night",
            "speed": 80,
            "hour": 2,
            "description": "Late-night highway — elevated risk",
        },
        "blackspot_danger": {
            "zone": "blackspot",
            "visibility": "fog",
            "speed": 60,
            "hour": 3,
            "description": "Blackspot + fog + 3 AM — maximum risk",
        },
        "school_rain": {
            "zone": "school_zone",
            "visibility": "rain",
            "speed": 30,
            "hour": 8,
            "description": "School zone in rain",
        },
    }

    def __init__(self) -> None:
        self.current: str = "normal"
        self._time_override: Optional[datetime] = None
        self._visibility_override: Optional[str] = None
        self._visibility_cycle = ["clear", "overcast", "rain", "fog", "night"]
        self._vis_idx = 0

    def switch(self, name: str) -> None:
        """Switch to a named scenario."""
        if name in self.SCENARIOS:
            self.current = name
            self._time_override = None
            self._visibility_override = None
            logger.info(f"Scenario: {name} — {self.SCENARIOS[name]['description']}")

    def set_day(self) -> None:
        """Override time to 14:00."""
        self._time_override = datetime.now().replace(hour=14, minute=0, second=0)

    def set_night(self) -> None:
        """Override time to 02:00."""
        self._time_override = datetime.now().replace(hour=2, minute=0, second=0)

    def cycle_visibility(self) -> str:
        """Cycle through visibility conditions. Returns new condition."""
        self._vis_idx = (self._vis_idx + 1) % len(self._visibility_cycle)
        self._visibility_override = self._visibility_cycle[self._vis_idx]
        logger.info(f"Visibility: {self._visibility_override}")
        return self._visibility_override

    def get_context(self, base_speed: Optional[float] = None) -> ContextState:
        """Build a ContextState for the current scenario + overrides."""
        s = self.SCENARIOS[self.current]
        if self._time_override:
            t = self._time_override
        else:
            t = datetime.now().replace(hour=s["hour"], minute=0, second=0)
        vis = self._visibility_override or s["visibility"]
        speed = base_speed if base_speed is not None else s["speed"]
        return ContextState(
            time_of_day=t,
            zone_type=s["zone"],
            visibility=vis,
            speed_kmh=speed,
        )

    @property
    def description(self) -> str:
        return self.SCENARIOS[self.current]["description"]


# ---------------------------------------------------------------------------
# The CRE
# ---------------------------------------------------------------------------

class ContextRiskEngine:
    """
    Weighted contextual risk fusion — the core of CAVSS.

    Args:
        config_path: Path to config.yaml.
        zone_manager: Optional pre-built ZoneManager.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        zone_manager: Optional[ZoneManager] = None,
    ) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        ce = cfg["context_engine"]
        self._weights: Dict[str, float] = ce["weights"]
        self._zone_mults: Dict[str, float] = ce["zone_multipliers"]
        self._vis_mults: Dict[str, float] = ce["visibility_multipliers"]
        self._speed_thresholds = ce["speed_thresholds"]
        self._alert_thresholds: Dict[str, float] = ce["alert_thresholds"]

        self._time_ctx = TimeContext(ce["time_multipliers"])
        self._zone_mgr = zone_manager

        logger.info("ContextRiskEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_risk(
        self, sensors: SensorReadings, context: ContextState
    ) -> RiskOutput:
        """
        Compute contextual risk score from sensor readings + context.

        Args:
            sensors: Normalised 0–1 values from DMS and ADAS.
            context: Current driving context.

        Returns:
            RiskOutput with score, alert level, breakdown, and messages.
        """
        # 1. Context multipliers
        time_mult = self._time_ctx.get_multiplier(context.time_of_day)
        zone_mult = self._zone_mults.get(context.zone_type, 1.0)
        vis_mult = self._vis_mults.get(context.visibility, 1.0)

        # Combined — cap at 2.5 to prevent runaway scores
        combined = min(time_mult * zone_mult * vis_mult, 2.5)

        # 2. Speed risk
        speed_risk = self._speed_to_risk(context.speed_kmh)

        # 3. Visibility risk (inverse of visibility clarity)
        vis_risk = 1.0 - self._visibility_clarity(context.visibility)

        # 4. Weighted component risks
        # DMS signals get the full context multiplier; ADAS / environment don't
        components: Dict[str, float] = {
            "drowsiness": sensors.drowsiness * self._weights["drowsiness"] * combined,
            "attention": sensors.attention * self._weights["attention"] * combined,
            "speed": speed_risk * self._weights["speed"],
            "lane_departure": sensors.lane_departure * self._weights["lane_departure"],
            "forward_collision": sensors.forward_collision * self._weights["forward_collision"],
            "visibility": vis_risk * self._weights["visibility"],
        }

        # 5. Sum and clamp
        raw_score = sum(components.values())
        final_score = float(max(0.0, min(1.0, raw_score)))

        # 6. Alert level
        level = self._get_alert_level(final_score)

        # 7. Dominant risk component
        dominant = max(components, key=components.get)

        # 8. Messages
        msg, voice = self._build_messages(level, dominant, context)

        return RiskOutput(
            final_risk_score=final_score,
            alert_level=level,
            component_risks=components,
            active_multipliers={
                "time": time_mult,
                "zone": zone_mult,
                "visibility": vis_mult,
                "combined": combined,
            },
            dominant_risk=dominant,
            alert_message=msg,
            voice_alert=voice,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _speed_to_risk(self, speed: float) -> float:
        for lo, hi, risk in self._speed_thresholds:
            if lo <= speed < hi:
                return float(risk)
        return 1.0

    def _visibility_clarity(self, vis: str) -> float:
        """Return 0–1 where 1 = perfectly clear."""
        table = {
            "clear": 1.0,
            "overcast": 0.8,
            "dusk_dawn": 0.7,
            "rain": 0.6,
            "night": 0.55,
            "heavy_rain": 0.4,
            "fog": 0.35,
        }
        return table.get(vis, 0.7)

    def _get_alert_level(self, score: float) -> str:
        thr = self._alert_thresholds
        if score >= thr["critical"]:
            return "critical"
        elif score >= thr["red"]:
            return "red"
        elif score >= thr["orange"]:
            return "orange"
        elif score >= thr["yellow"]:
            return "yellow"
        return "green"

    def _build_messages(
        self,
        level: str,
        dominant: str,
        context: ContextState,
    ) -> Tuple[Optional[str], Optional[str]]:
        if level == "green":
            return None, None

        # Context-aware suffixes
        time_sfx = ""
        h = context.time_of_day.hour
        if h < 6 or h >= 22:
            time_sfx = " It's late — consider resting."

        zone_sfx = ""
        if context.zone_type == "blackspot":
            zone_sfx = " You are in a high-accident zone."
        elif context.zone_type == "highway":
            zone_sfx = " Maintain safe highway distance."
        elif context.zone_type == "school_zone":
            zone_sfx = " School zone — watch for pedestrians."

        _msgs: Dict[str, Dict[str, str]] = {
            "drowsiness": {
                "yellow": f"Stay alert.{time_sfx}",
                "orange": f"Drowsiness detected. Take a break soon.{time_sfx}",
                "red": f"Warning: High drowsiness!{zone_sfx} Pull over when safe.",
                "critical": "CRITICAL: Too drowsy to drive. Pull over immediately.",
            },
            "attention": {
                "yellow": "Keep your eyes on the road.",
                "orange": "Distraction detected. Focus on driving.",
                "red": f"Warning: Extended distraction!{zone_sfx}",
                "critical": "CRITICAL: Stop driving distracted!",
            },
            "forward_collision": {
                "yellow": "Vehicle ahead — maintain distance.",
                "orange": "Caution: Close to vehicle ahead.",
                "red": "WARNING: Collision risk — slow down!",
                "critical": "BRAKE NOW!",
            },
            "lane_departure": {
                "yellow": "Drifting from lane.",
                "orange": "Lane departure warning.",
                "red": "WARNING: Leaving lane!",
                "critical": "CRITICAL: Severe lane departure!",
            },
            "speed": {
                "yellow": "High speed — stay alert.",
                "orange": "Speed is elevated for this zone.",
                "red": "WARNING: Too fast for conditions.",
                "critical": "CRITICAL: Reduce speed immediately.",
            },
        }

        msg = _msgs.get(dominant, {}).get(level, f"Alert: {level.upper()} risk.")
        voice = msg if level in ("orange", "red", "critical") else None
        return msg, voice
