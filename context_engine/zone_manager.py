"""
context_engine/zone_manager.py
Zone classification: detects blackspots, school zones, highway, urban etc.
Uses Haversine distance to match GPS coordinates against known zones.
"""

import json
import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

# Zone priority order (higher index = higher priority)
_ZONE_PRIORITY: Dict[str, int] = {
    "urban": 0,
    "residential": 1,
    "rural": 2,
    "highway": 3,
    "school_zone": 4,
    "blackspot": 5,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two GPS coordinates in kilometres.

    Args:
        lat1, lon1: First point in decimal degrees.
        lat2, lon2: Second point in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(max(0.0, 1.0 - a)))
    return R * c


class ZoneManager:
    """
    Classifies the current driving zone from GPS coordinates.

    Loads blackspot and school zone data from a JSON file.
    Falls back to graceful defaults when GPS or data is unavailable.

    Args:
        blackspot_file: Path to data/blackspots.json.
    """

    def __init__(self, blackspot_file: str = "data/blackspots.json") -> None:
        self.blackspots: List[Dict] = []
        self.school_zones: List[Dict] = []

        if os.path.exists(blackspot_file):
            try:
                with open(blackspot_file) as f:
                    data = json.load(f)
                self.blackspots = data.get("blackspots", [])
                self.school_zones = data.get("school_zones", [])
                logger.info(
                    f"ZoneManager: loaded {len(self.blackspots)} blackspots, "
                    f"{len(self.school_zones)} school zones"
                )
            except Exception as exc:
                logger.warning(f"Could not load blackspot file: {exc}")
        else:
            logger.warning(f"Blackspot file not found: {blackspot_file} — using defaults")

    def get_zone_type(self, lat: float, lon: float) -> str:
        """
        Determine zone type from GPS coordinates.

        Priority: blackspot > school_zone > highway > rural > urban

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Zone type string matching config.yaml zone_multipliers keys.
        """
        # Check blackspots (radius 0.5 km)
        for spot in self.blackspots:
            d = haversine_km(lat, lon, spot["latitude"], spot["longitude"])
            if d < 0.5:
                logger.debug(f"Blackspot: {spot.get('name', 'unknown')} ({d:.2f} km)")
                return "blackspot"

        # Check school zones (radius 0.3 km)
        for zone in self.school_zones:
            d = haversine_km(lat, lon, zone["latitude"], zone["longitude"])
            if d < 0.3:
                logger.debug(f"School zone: {zone.get('name', 'unknown')} ({d:.2f} km)")
                return "school_zone"

        # Default: urban (demo route will override this via scenario simulator)
        return "urban"

    def get_nearest_blackspot(
        self, lat: float, lon: float
    ) -> Optional[Dict]:
        """Return the nearest blackspot entry and its distance, or None."""
        if not self.blackspots:
            return None
        nearest = min(
            self.blackspots,
            key=lambda s: haversine_km(lat, lon, s["latitude"], s["longitude"]),
        )
        nearest["distance_km"] = haversine_km(lat, lon, nearest["latitude"], nearest["longitude"])
        return nearest
