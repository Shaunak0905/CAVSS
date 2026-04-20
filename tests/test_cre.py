"""
tests/test_cre.py
Unit tests for the Context Risk Engine.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from context_engine.risk_calculator import ContextRiskEngine, SensorReadings, ContextState


CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def make_cre() -> ContextRiskEngine:
    return ContextRiskEngine(config_path=CONFIG)


def test_normal_driving_is_green() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=0.05, attention=0.05, lane_departure=0.0, forward_collision=0.0)
    context = ContextState(
        time_of_day=datetime(2026, 3, 28, 14, 0),
        zone_type="urban", visibility="clear", speed_kmh=40,
    )
    result = cre.calculate_risk(sensors, context)
    assert result.alert_level == "green", f"Expected green, got {result.alert_level} (score={result.final_risk_score:.3f})"
    print(f"[PASS] Normal driving → green ({result.final_risk_score:.3f})")


def test_max_danger_is_critical_or_red() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=0.9, attention=0.8, lane_departure=0.5, forward_collision=0.8)
    context = ContextState(
        time_of_day=datetime(2026, 3, 28, 3, 0),
        zone_type="blackspot", visibility="fog", speed_kmh=100,
    )
    result = cre.calculate_risk(sensors, context)
    assert result.alert_level in ("red", "critical"), \
        f"Expected red/critical, got {result.alert_level} (score={result.final_risk_score:.3f})"
    print(f"[PASS] Max danger → {result.alert_level} ({result.final_risk_score:.3f})")


def test_score_bounded_0_1() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=1.0, attention=1.0, lane_departure=1.0, forward_collision=1.0)
    context = ContextState(
        time_of_day=datetime(2026, 3, 28, 2, 30),
        zone_type="blackspot", visibility="fog", speed_kmh=120,
    )
    result = cre.calculate_risk(sensors, context)
    assert 0.0 <= result.final_risk_score <= 1.0, f"Score out of bounds: {result.final_risk_score}"
    print(f"[PASS] Score bounded: {result.final_risk_score:.3f}")


def test_night_multiplier_higher_than_day() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=0.4, attention=0.3, lane_departure=0.0, forward_collision=0.0)
    context_day = ContextState(
        time_of_day=datetime(2026, 3, 28, 14, 0),
        zone_type="urban", visibility="clear", speed_kmh=60,
    )
    context_night = ContextState(
        time_of_day=datetime(2026, 3, 28, 2, 0),
        zone_type="urban", visibility="clear", speed_kmh=60,
    )
    r_day = cre.calculate_risk(sensors, context_day)
    r_night = cre.calculate_risk(sensors, context_night)
    assert r_night.final_risk_score > r_day.final_risk_score, \
        f"Night risk ({r_night.final_risk_score:.3f}) should exceed day ({r_day.final_risk_score:.3f})"
    print(f"[PASS] Night > Day: {r_night.final_risk_score:.3f} > {r_day.final_risk_score:.3f}")


def test_blackspot_multiplier_higher_than_urban() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=0.4, attention=0.3, lane_departure=0.0, forward_collision=0.0)
    context_urban = ContextState(
        time_of_day=datetime(2026, 3, 28, 14, 0),
        zone_type="urban", visibility="clear", speed_kmh=60,
    )
    context_blackspot = ContextState(
        time_of_day=datetime(2026, 3, 28, 14, 0),
        zone_type="blackspot", visibility="clear", speed_kmh=60,
    )
    r_urban = cre.calculate_risk(sensors, context_urban)
    r_bs = cre.calculate_risk(sensors, context_blackspot)
    assert r_bs.final_risk_score > r_urban.final_risk_score, \
        f"Blackspot ({r_bs.final_risk_score:.3f}) should exceed urban ({r_urban.final_risk_score:.3f})"
    print(f"[PASS] Blackspot > Urban: {r_bs.final_risk_score:.3f} > {r_urban.final_risk_score:.3f}")


def test_component_breakdown_sums_to_final() -> None:
    cre = make_cre()
    sensors = SensorReadings(drowsiness=0.5, attention=0.3, lane_departure=0.1, forward_collision=0.2)
    context = ContextState(
        time_of_day=datetime(2026, 3, 28, 18, 0),
        zone_type="highway", visibility="overcast", speed_kmh=80,
    )
    result = cre.calculate_risk(sensors, context)
    component_sum = sum(result.component_risks.values())
    assert abs(component_sum - result.final_risk_score) < 1e-6 or result.final_risk_score == 1.0, \
        f"Component sum {component_sum:.4f} doesn't match final {result.final_risk_score:.4f}"
    print(f"[PASS] Component sum consistent: {component_sum:.4f}")


if __name__ == "__main__":
    print("Running CRE tests...\n")
    test_normal_driving_is_green()
    test_max_danger_is_critical_or_red()
    test_score_bounded_0_1()
    test_night_multiplier_higher_than_day()
    test_blackspot_multiplier_higher_than_urban()
    test_component_breakdown_sums_to_final()
    print("\nAll tests passed!")
