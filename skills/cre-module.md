---
name: cre-module
description: |
  Build the Context Risk Engine - the core innovation of CAVSS. Use this skill when working on:
  - Risk score calculation and fusion
  - Context multipliers (time, zone, visibility)
  - Alert level determination
  - Weight optimization
  Trigger on: "CRE", "context engine", "risk calculation", "risk fusion", "weighted risk", "context multiplier", "alert threshold", "risk score"
---

# Context Risk Engine (CRE) Development Skill

## Overview

The CRE is what makes CAVSS different from every other drowsiness detector. It doesn't just detect individual risks — it **contextualizes** them.

**Core Principle**: The same sensor reading produces different risk scores based on WHERE, WHEN, and HOW the person is driving.

## The Formula

```
Final_Risk = Σ(Wi × Ci × Si)

Where:
- Wi = Base weight for parameter i
- Ci = Context multiplier (product of all applicable contexts)
- Si = Sensor reading (normalized 0-1)
```

## Implementation

### Main Risk Calculator

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import yaml

@dataclass
class ContextState:
    """Current driving context."""
    time_of_day: datetime
    zone_type: str  # "urban", "highway", "blackspot", "school", "rural"
    visibility: str  # "clear", "overcast", "rain", "fog", "night"
    speed_kmh: float
    location: Optional[Dict] = None  # GPS coordinates

@dataclass
class SensorReadings:
    """Normalized sensor readings (0-1)."""
    drowsiness: float
    attention: float
    lane_departure: float
    forward_collision: float
    # Speed is context, not sensor

@dataclass
class RiskOutput:
    """Output from CRE."""
    final_risk_score: float  # 0-1
    alert_level: str  # "green", "yellow", "orange", "red", "critical"
    
    # Breakdown for debugging/display
    component_risks: Dict[str, float]
    active_multipliers: Dict[str, float]
    dominant_risk: str
    
    # Recommendations
    alert_message: Optional[str]
    voice_alert: Optional[str]

class ContextRiskEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.weights = config["context_engine"]["weights"]
        self.time_multipliers = config["context_engine"]["time_multipliers"]
        self.zone_multipliers = config["context_engine"]["zone_multipliers"]
        self.visibility_multipliers = config["context_engine"]["visibility_multipliers"]
        self.speed_thresholds = config["context_engine"]["speed_thresholds"]
        self.alert_thresholds = config["context_engine"]["alert_thresholds"]
    
    def calculate_risk(
        self, 
        sensors: SensorReadings, 
        context: ContextState
    ) -> RiskOutput:
        """
        Main risk calculation function.
        
        This is the core innovation - contextual fusion.
        """
        # Step 1: Get context multipliers
        time_mult = self._get_time_multiplier(context.time_of_day)
        zone_mult = self._get_zone_multiplier(context.zone_type)
        visibility_mult = self._get_visibility_multiplier(context.visibility)
        
        # Combined context multiplier (product, not sum)
        # Cap at 2.5 to prevent runaway scores
        combined_context = min(time_mult * zone_mult * visibility_mult, 2.5)
        
        # Step 2: Get speed risk
        speed_risk = self._get_speed_risk(context.speed_kmh)
        
        # Step 3: Calculate weighted component risks
        component_risks = {
            "drowsiness": sensors.drowsiness * self.weights["drowsiness"] * combined_context,
            "attention": sensors.attention * self.weights["attention"] * combined_context,
            "speed": speed_risk * self.weights["speed"],
            "lane_departure": sensors.lane_departure * self.weights["lane_departure"],
            "forward_collision": sensors.forward_collision * self.weights["forward_collision"],
            "visibility": (1 - self._visibility_to_score(context.visibility)) * self.weights["visibility"]
        }
        
        # Step 4: Sum all risks
        final_risk = sum(component_risks.values())
        
        # Clamp to 0-1
        final_risk = max(0.0, min(1.0, final_risk))
        
        # Step 5: Determine alert level
        alert_level = self._get_alert_level(final_risk)
        
        # Step 6: Find dominant risk
        dominant_risk = max(component_risks, key=component_risks.get)
        
        # Step 7: Generate alert message
        alert_message, voice_alert = self._generate_alerts(
            alert_level, dominant_risk, sensors, context
        )
        
        return RiskOutput(
            final_risk_score=final_risk,
            alert_level=alert_level,
            component_risks=component_risks,
            active_multipliers={
                "time": time_mult,
                "zone": zone_mult,
                "visibility": visibility_mult,
                "combined": combined_context
            },
            dominant_risk=dominant_risk,
            alert_message=alert_message,
            voice_alert=voice_alert
        )
    
    def _get_time_multiplier(self, time: datetime) -> float:
        """Get risk multiplier based on time of day."""
        hour = time.hour
        
        for start, end, mult in self.time_multipliers:
            if start <= hour < end:
                return mult
        
        return 1.0  # Default
    
    def _get_zone_multiplier(self, zone: str) -> float:
        """Get risk multiplier based on zone type."""
        return self.zone_multipliers.get(zone, 1.0)
    
    def _get_visibility_multiplier(self, visibility: str) -> float:
        """Get risk multiplier based on visibility conditions."""
        return self.visibility_multipliers.get(visibility, 1.0)
    
    def _visibility_to_score(self, visibility: str) -> float:
        """Convert visibility condition to 0-1 score (1 = clear)."""
        scores = {
            "clear": 1.0,
            "overcast": 0.8,
            "rain": 0.6,
            "fog": 0.4,
            "night": 0.5
        }
        return scores.get(visibility, 0.7)
    
    def _get_speed_risk(self, speed_kmh: float) -> float:
        """Convert speed to risk score."""
        for min_speed, max_speed, risk in self.speed_thresholds:
            if min_speed <= speed_kmh < max_speed:
                return risk
        return 1.0  # Very high speed
    
    def _get_alert_level(self, risk_score: float) -> str:
        """Determine alert level from risk score."""
        if risk_score >= self.alert_thresholds["critical"]:
            return "critical"
        elif risk_score >= self.alert_thresholds["red"]:
            return "red"
        elif risk_score >= self.alert_thresholds["orange"]:
            return "orange"
        elif risk_score >= self.alert_thresholds["yellow"]:
            return "yellow"
        else:
            return "green"
    
    def _generate_alerts(
        self, 
        level: str, 
        dominant: str, 
        sensors: SensorReadings,
        context: ContextState
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate appropriate alert messages."""
        
        if level == "green":
            return None, None
        
        # Context-aware messages
        time_suffix = ""
        if context.time_of_day.hour < 6 or context.time_of_day.hour >= 22:
            time_suffix = " It's late - consider stopping for rest."
        
        zone_suffix = ""
        if context.zone_type == "blackspot":
            zone_suffix = " You're in a high-accident zone."
        elif context.zone_type == "highway":
            zone_suffix = " Maintain safe highway distance."
        
        messages = {
            "drowsiness": {
                "yellow": f"Stay alert.{time_suffix}",
                "orange": f"Drowsiness detected. Take a break soon.{time_suffix}",
                "red": f"Warning: High drowsiness.{zone_suffix} Pull over when safe.",
                "critical": "CRITICAL: Pull over immediately. You're too drowsy to drive."
            },
            "attention": {
                "yellow": "Keep your eyes on the road.",
                "orange": "Distraction detected. Focus on driving.",
                "red": f"Warning: Extended distraction.{zone_suffix}",
                "critical": "CRITICAL: Stop driving distracted!"
            },
            "forward_collision": {
                "yellow": "Vehicle ahead - maintain distance.",
                "orange": "Caution: Close to vehicle ahead.",
                "red": "WARNING: Collision risk - slow down!",
                "critical": "BRAKE NOW!"
            },
            "lane_departure": {
                "yellow": "Drifting from lane.",
                "orange": "Lane departure warning.",
                "red": "WARNING: Leaving lane!",
                "critical": "CRITICAL: Lane departure!"
            }
        }
        
        msg = messages.get(dominant, {}).get(level, f"Alert level: {level}")
        
        # Voice alert only for orange and above
        voice = msg if level in ["orange", "red", "critical"] else None
        
        return msg, voice
```

## Zone Management

```python
import json
from math import radians, sin, cos, sqrt, atan2

class ZoneManager:
    def __init__(self, blackspot_file: str = "data/blackspots.json"):
        with open(blackspot_file) as f:
            data = json.load(f)
        self.blackspots = data.get("blackspots", [])
        self.school_zones = data.get("school_zones", [])
    
    def get_zone_type(self, lat: float, lon: float) -> str:
        """
        Determine zone type based on coordinates.
        
        Priority: blackspot > school > highway > rural > urban
        """
        # Check blackspots first (highest priority)
        for spot in self.blackspots:
            if self._distance_km(lat, lon, spot["latitude"], spot["longitude"]) < 0.5:
                return "blackspot"
        
        # Check school zones
        for zone in self.school_zones:
            if self._distance_km(lat, lon, zone["latitude"], zone["longitude"]) < 0.3:
                return "school_zone"
        
        # Default logic based on speed/road type
        # (In real implementation, use road data)
        return "urban"  # Default for demo
    
    def _distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula for distance between two points."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
```

## Time Context

```python
from datetime import datetime

class TimeContext:
    """Manage time-based risk adjustments."""
    
    @staticmethod
    def get_time_category(time: datetime) -> str:
        """Categorize time for risk assessment."""
        hour = time.hour
        
        if 0 <= hour < 5:
            return "night_peak"  # Highest drowsiness risk
        elif 5 <= hour < 7:
            return "morning_early"
        elif 7 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "afternoon_early"  # Post-lunch dip
        elif 14 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night_late"
    
    @staticmethod
    def get_fatigue_baseline(time: datetime) -> float:
        """
        Get baseline fatigue expectation based on circadian rhythm.
        
        Returns 0-1 where higher = more expected fatigue.
        """
        hour = time.hour
        
        # Circadian low points: 2-4 AM and 2-4 PM
        if 2 <= hour <= 4:
            return 0.8  # High fatigue expected
        elif 14 <= hour <= 16:
            return 0.4  # Moderate (post-lunch)
        elif 22 <= hour or hour < 2:
            return 0.6  # Elevated
        elif 4 < hour < 7:
            return 0.3  # Rising alertness
        else:
            return 0.2  # Normal alertness
```

## Module Structure

```
context_engine/
├── __init__.py
├── risk_calculator.py    # Main CRE logic
├── zone_manager.py       # Blackspot/zone detection
├── time_context.py       # Time-based adjustments
├── weather_context.py    # Visibility factors
└── scenario_simulator.py # Demo scenario control
```

## Demo Scenarios

```python
class ScenarioSimulator:
    """Pre-defined scenarios for demo."""
    
    SCENARIOS = {
        "normal_city": {
            "zone": "urban",
            "visibility": "clear",
            "speed": 40,
            "time": datetime(2026, 3, 28, 14, 0),  # 2 PM
            "description": "Normal daytime city driving"
        },
        "highway_night": {
            "zone": "highway",
            "visibility": "night",
            "speed": 80,
            "time": datetime(2026, 3, 28, 2, 0),  # 2 AM
            "description": "Late night highway driving"
        },
        "blackspot_drowsy": {
            "zone": "blackspot",
            "visibility": "fog",
            "speed": 60,
            "time": datetime(2026, 3, 28, 3, 30),  # 3:30 AM
            "description": "Dangerous conditions - maximum risk"
        },
        "school_zone_rain": {
            "zone": "school_zone",
            "visibility": "rain",
            "speed": 30,
            "time": datetime(2026, 3, 28, 8, 0),  # 8 AM
            "description": "School zone in rain"
        }
    }
    
    def __init__(self):
        self.current_scenario = "normal_city"
    
    def get_context(self) -> ContextState:
        """Get context for current scenario."""
        s = self.SCENARIOS[self.current_scenario]
        return ContextState(
            time_of_day=s["time"],
            zone_type=s["zone"],
            visibility=s["visibility"],
            speed_kmh=s["speed"]
        )
    
    def switch_scenario(self, name: str):
        """Switch to different scenario."""
        if name in self.SCENARIOS:
            self.current_scenario = name
```

## Output Visualization

```python
def format_risk_display(output: RiskOutput) -> str:
    """Format risk output for dashboard display."""
    
    # Color coding
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "orange": "\033[33m",
        "red": "\033[91m",
        "critical": "\033[91m\033[1m"
    }
    reset = "\033[0m"
    
    display = f"""
╔══════════════════════════════════════════════════════════╗
║  RISK ASSESSMENT: {colors[output.alert_level]}{output.alert_level.upper():^10}{reset}  Score: {output.final_risk_score:.2f}  ║
╠══════════════════════════════════════════════════════════╣
║  Component Breakdown:                                    ║
║    Drowsiness:        {output.component_risks['drowsiness']:.3f}                          ║
║    Attention:         {output.component_risks['attention']:.3f}                          ║
║    Speed:             {output.component_risks['speed']:.3f}                          ║
║    Lane Departure:    {output.component_risks['lane_departure']:.3f}                          ║
║    Forward Collision: {output.component_risks['forward_collision']:.3f}                          ║
╠══════════════════════════════════════════════════════════╣
║  Active Multipliers:                                     ║
║    Time:       {output.active_multipliers['time']:.1f}x                                  ║
║    Zone:       {output.active_multipliers['zone']:.1f}x                                  ║
║    Visibility: {output.active_multipliers['visibility']:.1f}x                                  ║
║    Combined:   {output.active_multipliers['combined']:.2f}x                                 ║
╠══════════════════════════════════════════════════════════╣
║  Dominant Risk: {output.dominant_risk.upper():<40} ║
╚══════════════════════════════════════════════════════════╝
"""
    
    if output.alert_message:
        display += f"\n⚠️  {output.alert_message}"
    
    return display
```

## Testing the CRE

```python
def test_cre_scenarios():
    """Test CRE with various scenarios."""
    cre = ContextRiskEngine()
    
    # Scenario 1: Normal driving
    sensors_normal = SensorReadings(
        drowsiness=0.1, attention=0.1, 
        lane_departure=0.0, forward_collision=0.0
    )
    context_normal = ContextState(
        time_of_day=datetime(2026, 3, 28, 14, 0),
        zone_type="urban", visibility="clear", speed_kmh=40
    )
    result = cre.calculate_risk(sensors_normal, context_normal)
    assert result.alert_level == "green", f"Expected green, got {result.alert_level}"
    
    # Scenario 2: Drowsy on highway at night near blackspot
    sensors_drowsy = SensorReadings(
        drowsiness=0.7, attention=0.3,
        lane_departure=0.2, forward_collision=0.1
    )
    context_dangerous = ContextState(
        time_of_day=datetime(2026, 3, 28, 2, 0),
        zone_type="blackspot", visibility="fog", speed_kmh=80
    )
    result = cre.calculate_risk(sensors_drowsy, context_dangerous)
    assert result.alert_level in ["red", "critical"], f"Expected red/critical, got {result.alert_level}"
    
    print("All CRE tests passed!")
```
