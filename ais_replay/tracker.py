"""
Per-vessel track store with dead-reckoning between sparse AIS updates.

AIS position reports arrive every 2-30+ s per vessel (class/speed
dependent), with dropouts; the automaton wants a complete obstacle list
(x, y, v, psi) every tick. This tracker is the first layer of the
sensor-noise plan (HANDOFF §4): it absorbs the sparseness so the core
keeps consuming clean per-tick states.

Dead-reckoning is constant-velocity from the last report (the same
motion model the unsafe-set API assumes). Upgrading a track to an EKF
happens here, behind the same obstacles_at() interface, when real data
shows it is needed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ais_replay.geo import LocalFrame, cog_to_psi, knots_to_ms

# AIS sentinel values
_COG_UNAVAILABLE = 360.0
_SOG_UNAVAILABLE = 102.3  # knots, per ITU-R M.1371


@dataclass
class AISReport:
    """One decoded AIS position report (raw nautical units)."""
    mmsi: int
    t: float           # epoch seconds
    lat: float
    lon: float
    sog_knots: float
    cog_deg: float
    name: str = ""


@dataclass
class _Track:
    x: float
    y: float
    v: float           # m/s
    psi: float         # rad, CCW from +x
    t_last: float      # epoch seconds of last report
    name: str = ""


class TrafficTracker:
    """
    Ingests AISReports, emits dead-reckoned (x, y, v, psi) obstacle
    tuples for any query time.
    """

    def __init__(self, frame: LocalFrame, max_age: float = 300.0):
        """
        Args:
            frame: lat/lon projection for the operating area
            max_age: drop a track this many seconds after its last
                report (default 5 min — beyond that dead-reckoning is
                guesswork and the vessel has likely left the area)
        """
        self.frame = frame
        self.max_age = max_age
        self._tracks: Dict[int, _Track] = {}

    def ingest(self, report: AISReport) -> None:
        """Insert/update a vessel track from a position report."""
        existing = self._tracks.get(report.mmsi)
        if existing is not None and report.t < existing.t_last:
            return  # stale out-of-order report

        x, y = self.frame.to_xy(report.lat, report.lon)

        sog = report.sog_knots
        if sog is None or sog >= _SOG_UNAVAILABLE:
            sog = 0.0
        cog = report.cog_deg
        if cog is None or cog >= _COG_UNAVAILABLE:
            # Course unavailable: keep previous heading if we have one
            psi = existing.psi if existing is not None else 0.0
        else:
            psi = float(cog_to_psi(cog))

        self._tracks[report.mmsi] = _Track(
            x=x, y=y,
            v=float(knots_to_ms(sog)),
            psi=psi,
            t_last=report.t,
            name=report.name or (existing.name if existing else ""),
        )

    def obstacles_at(
        self,
        t: float,
        near: Optional[Tuple[float, float]] = None,
        within: float = 0.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Dead-reckoned obstacle list at time t, in the automaton's
        (x, y, v, psi) tuple format. Expires stale tracks as a side
        effect.

        Args:
            near, within: when both given, only tracks within `within`
                metres of `near` are returned. Dense areas (a strait
                bbox easily holds 100+ vessels) overwhelm the guards'
                per-obstacle hull computations; real systems filter to
                sensor/relevance range the same way.
        """
        out = []
        for mmsi in list(self._tracks):
            trk = self._tracks[mmsi]
            age = t - trk.t_last
            if age > self.max_age:
                del self._tracks[mmsi]
                continue
            dt = max(age, 0.0)
            x = trk.x + trk.v * np.cos(trk.psi) * dt
            y = trk.y + trk.v * np.sin(trk.psi) * dt
            if near is not None and within > 0.0:
                if np.hypot(x - near[0], y - near[1]) > within:
                    continue
            out.append((x, y, trk.v, trk.psi))
        return out

    def track_names(self) -> Dict[int, str]:
        return {mmsi: trk.name for mmsi, trk in self._tracks.items()}

    def __len__(self) -> int:
        return len(self._tracks)
