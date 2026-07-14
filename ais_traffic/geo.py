"""
Lat/lon <-> local metric frame conversion.

The automaton works in a flat x/y metre frame (x = east, y = north,
headings in radians CCW from +x). AIS positions are WGS84 lat/lon.
An equirectangular projection around a fixed origin is accurate to
well under 0.1% for the harbour/strait scales this adapter targets
(tens of km) — far below AIS position noise.
"""

from typing import Tuple

import numpy as np

# Metres per degree of latitude (WGS84 mean)
M_PER_DEG_LAT = 111_320.0


class LocalFrame:
    """Equirectangular projection centred on (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(lat0))

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """WGS84 lat/lon -> local (x east, y north) in metres."""
        x = (lon - self.lon0) * self._m_per_deg_lon
        y = (lat - self.lat0) * M_PER_DEG_LAT
        return (x, y)

    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Local (x, y) metres -> WGS84 lat/lon."""
        lat = self.lat0 + y / M_PER_DEG_LAT
        lon = self.lon0 + x / self._m_per_deg_lon
        return (lat, lon)


def cog_to_psi(cog_deg: float) -> float:
    """
    AIS course-over-ground (degrees clockwise from true north) to the
    automaton's heading convention (radians CCW from +x/east).
    """
    return np.radians(90.0 - cog_deg)


def knots_to_ms(sog_knots: float) -> float:
    """Speed over ground: knots -> m/s."""
    return sog_knots * 0.514444
