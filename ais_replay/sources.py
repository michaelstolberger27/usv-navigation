"""
AIS data sources: recorded JSONL replay and the aisstream.io live feed.

The recorded format is one raw aisstream.io message JSON object per
line (exactly what scripts/record_ais.py captures), so a recording can
be replayed bit-identically. parse_aisstream_message() is shared by
both paths.
"""

import json
from datetime import datetime
from typing import Iterator, List, Optional

from ais_replay.tracker import AISReport, TrafficTracker


def _parse_time_utc(value) -> Optional[float]:
    """aisstream.io time_utc -> epoch seconds (best effort)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # e.g. "2024-01-21 07:01:02.123456789 +0000 UTC"
    text = str(value).replace(" UTC", "")
    # Trim sub-microsecond digits that datetime cannot parse
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        for candidate in (text, _trim_fractional(text)):
            try:
                return datetime.strptime(candidate, fmt).timestamp()
            except ValueError:
                continue
    return None


def _trim_fractional(text: str) -> str:
    """Truncate fractional seconds to 6 digits for %f parsing."""
    if "." not in text:
        return text
    head, _, tail = text.partition(".")
    digits = ""
    rest = tail
    for i, ch in enumerate(tail):
        if ch.isdigit():
            digits += ch
        else:
            rest = tail[i:]
            break
    else:
        rest = ""
    return f"{head}.{digits[:6]}{rest}"


def parse_aisstream_message(msg: dict) -> Optional[AISReport]:
    """
    Decode one aisstream.io message dict into an AISReport.

    Returns None for anything that is not a usable position report.
    """
    if msg.get("MessageType") != "PositionReport":
        return None

    meta = msg.get("MetaData", {})
    body = msg.get("Message", {}).get("PositionReport", {})

    lat = body.get("Latitude", meta.get("latitude"))
    lon = body.get("Longitude", meta.get("longitude"))
    mmsi = meta.get("MMSI", body.get("UserID"))
    t = _parse_time_utc(meta.get("time_utc"))
    if lat is None or lon is None or mmsi is None or t is None:
        return None

    return AISReport(
        mmsi=int(mmsi),
        t=t,
        lat=float(lat),
        lon=float(lon),
        sog_knots=float(body.get("Sog", 0.0)),
        cog_deg=float(body.get("Cog", 360.0)),
        name=str(meta.get("ShipName", "")).strip(),
    )


class RecordedAISSource:
    """
    Replays a JSONL recording (one aisstream.io message per line).

    Reports are sorted by timestamp on load; feed_until() pushes them
    into a TrafficTracker as replay time advances.
    """

    def __init__(self, path: str):
        self.reports: List[AISReport] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                report = parse_aisstream_message(json.loads(line))
                if report is not None:
                    self.reports.append(report)
        self.reports.sort(key=lambda r: r.t)
        self._cursor = 0
        if not self.reports:
            raise ValueError(f"No usable PositionReports in {path}")

    @property
    def t_start(self) -> float:
        return self.reports[0].t

    @property
    def t_end(self) -> float:
        return self.reports[-1].t

    def feed_until(self, tracker: TrafficTracker, t: float) -> int:
        """Ingest all reports with timestamp <= t. Returns count fed."""
        fed = 0
        while self._cursor < len(self.reports) and self.reports[self._cursor].t <= t:
            tracker.ingest(self.reports[self._cursor])
            self._cursor += 1
            fed += 1
        return fed

    def rewind(self) -> None:
        self._cursor = 0

    def __iter__(self) -> Iterator[AISReport]:
        return iter(self.reports)


class AISStreamSource:
    """
    Live aisstream.io websocket feed (free API key: https://aisstream.io).

    Runs a background thread that pushes decoded reports into a
    TrafficTracker as they arrive. Requires the `websockets` package
    (`pip install -e .[ais]`).

    Note: built to the aisstream.io v0 message format; live operation
    needs an API key and network access, so this class is exercised
    manually rather than by the test suite.
    """

    URL = "wss://stream.aisstream.io/v0/stream"

    def __init__(self, api_key: str, bbox: list, tracker: TrafficTracker):
        """
        Args:
            api_key: aisstream.io API key
            bbox: [[lat_min, lon_min], [lat_max, lon_max]]
            tracker: tracker to feed (thread-safe enough: dict updates
                are atomic under the GIL and consumers tolerate tearing
                by one report)
        """
        self.api_key = api_key
        self.bbox = bbox
        self.tracker = tracker
        self._thread = None
        self._stop = False

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def _run(self) -> None:
        import asyncio
        asyncio.run(self._stream())

    async def _stream(self) -> None:
        import websockets  # lazy: only live mode needs it

        subscription = json.dumps({
            "APIKey": self.api_key,
            "BoundingBoxes": [self.bbox],
            "FilterMessageTypes": ["PositionReport"],
        })
        async with websockets.connect(self.URL) as ws:
            await ws.send(subscription)
            async for raw in ws:
                if self._stop:
                    break
                try:
                    report = parse_aisstream_message(json.loads(raw))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if report is not None:
                    self.tracker.ingest(report)
