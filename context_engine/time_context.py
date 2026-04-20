"""
context_engine/time_context.py
Time-of-day risk multipliers and fatigue baseline based on circadian rhythm.
"""

import logging
from datetime import datetime
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TimeContext:
    """
    Computes time-based risk adjustments.

    Multipliers are loaded from config.yaml so they can be tuned without
    code changes.

    Args:
        time_multipliers: List of [start_hour, end_hour, multiplier] from config.
    """

    def __init__(self, time_multipliers: List[List]) -> None:
        # Each entry: [start_hour, end_hour, multiplier]
        self._multipliers: List[Tuple[int, int, float]] = [
            (int(entry[0]), int(entry[1]), float(entry[2]))
            for entry in time_multipliers
        ]
        logger.debug(f"TimeContext: {len(self._multipliers)} time bands loaded")

    def get_multiplier(self, dt: datetime) -> float:
        """
        Return the time-of-day risk multiplier for a given datetime.

        Args:
            dt: The datetime to evaluate.

        Returns:
            Multiplier (1.0 = baseline, 1.5 = peak risk at 2–5 AM).
        """
        hour = dt.hour
        for start, end, mult in self._multipliers:
            if start <= hour < end:
                return mult
        return 1.0

    @staticmethod
    def get_time_category(dt: datetime) -> str:
        """
        Categorise time for display / logging purposes.

        Returns:
            One of: night_peak, morning_early, morning, afternoon_early,
                    afternoon, evening, night_late.
        """
        h = dt.hour
        if 0 <= h < 5:
            return "night_peak"
        elif 5 <= h < 7:
            return "morning_early"
        elif 7 <= h < 12:
            return "morning"
        elif 12 <= h < 14:
            return "afternoon_early"
        elif 14 <= h < 18:
            return "afternoon"
        elif 18 <= h < 22:
            return "evening"
        else:
            return "night_late"

    @staticmethod
    def get_fatigue_baseline(dt: datetime) -> float:
        """
        Estimate the baseline fatigue level from the circadian rhythm.

        Returns:
            Float 0–1 (higher = more fatigue expected).
        """
        h = dt.hour
        if 2 <= h <= 4:
            return 0.8   # Peak drowsiness window
        elif 14 <= h <= 16:
            return 0.4   # Post-lunch dip
        elif h >= 22 or h < 2:
            return 0.6   # Late night / very early
        elif 4 < h < 7:
            return 0.3   # Rising alertness
        else:
            return 0.2   # Normal daytime alertness
