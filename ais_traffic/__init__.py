"""
AIS replay adapter — feeds real ship traffic into the COLAV automaton.

Mirrors the role of commonocean_integration/ for a different data source:
everything that knows about AIS message formats, lat/lon geodesy, or
network streams lives here, never in src/colav_automaton/.

Components:
    geo       — lat/lon <-> local metric frame (equirectangular)
    tracker   — per-vessel track store with dead-reckoning between
                sparse AIS updates (sensor-noise plan, point 1)
    sources   — RecordedAISSource (JSONL replay), AISStreamSource
                (live aisstream.io websocket; requires API key)
    runner    — drives the automaton through replayed/live traffic
                using the same pattern as the CommonOcean adapter
"""

from ais_traffic.geo import LocalFrame
from ais_traffic.tracker import AISReport, TrafficTracker
from ais_traffic.sources import RecordedAISSource, parse_aisstream_message

__all__ = [
    "LocalFrame",
    "AISReport",
    "TrafficTracker",
    "RecordedAISSource",
    "parse_aisstream_message",
]
