"""
feeds/mock_gps.py
Simulated GPS feed — replays a pre-recorded Pune route for demo purposes.
"""

import json
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MockGPS:
    """
    Simulates GPS movement along a pre-defined route.

    Interpolates between waypoints based on elapsed time.
    Used by the CRE to determine zone type and speed context.

    Args:
        route_file: Path to data/demo_route.json.
        loop: Restart from beginning when route ends.
    """

    def __init__(self, route_file: str = "data/demo_route.json", loop: bool = True) -> None:
        self._waypoints: list = []
        self._loop = loop
        self._start_time: Optional[float] = None

        try:
            with open(route_file) as f:
                data = json.load(f)
            self._waypoints = data.get("waypoints", [])
            logger.info(f"MockGPS: loaded {len(self._waypoints)} waypoints from {route_file}")
        except Exception as exc:
            logger.warning(f"MockGPS: cannot load route ({exc}) — using static Pune location")
            self._waypoints = [
                {"time_seconds": 0, "lat": 18.5204, "lon": 73.8567, "zone": "urban",
                 "speed_kmh": 40, "name": "Pune City"}
            ]

    def start(self) -> None:
        """Begin playback."""
        self._start_time = time.time()

    def get_position(self) -> Dict:
        """
        Return current simulated GPS position.

        Returns:
            Dict with keys: lat, lon, zone, speed_kmh, name, elapsed_s
        """
        if self._start_time is None:
            self.start()

        elapsed = time.time() - self._start_time
        total_time = self._waypoints[-1]["time_seconds"] if self._waypoints else 1

        if self._loop and total_time > 0:
            elapsed = elapsed % total_time

        # Find surrounding waypoints
        for i in range(len(self._waypoints) - 1):
            w0 = self._waypoints[i]
            w1 = self._waypoints[i + 1]
            if w0["time_seconds"] <= elapsed < w1["time_seconds"]:
                # Linear interpolation
                t_frac = (elapsed - w0["time_seconds"]) / max(
                    w1["time_seconds"] - w0["time_seconds"], 1
                )
                return {
                    "lat": w0["lat"] + t_frac * (w1["lat"] - w0["lat"]),
                    "lon": w0["lon"] + t_frac * (w1["lon"] - w0["lon"]),
                    "zone": w0["zone"],
                    "speed_kmh": w0["speed_kmh"] + t_frac * (w1["speed_kmh"] - w0["speed_kmh"]),
                    "name": w0["name"],
                    "elapsed_s": elapsed,
                }

        # Past last waypoint
        last = self._waypoints[-1]
        return {**last, "elapsed_s": elapsed}
