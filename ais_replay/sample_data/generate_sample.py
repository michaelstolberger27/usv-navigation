"""
Generate sample_strait.jsonl — a synthetic AIS recording in the exact
aisstream.io message format, so the replay path is bit-identical to a
real recording (see scripts/record_ais.py for capturing real data).

Scenario (Singapore Strait coordinates, ~1.2N 103.85E):
  - ego transit is intended west -> east (~4 km)
  - LADY MARGAUX:  westbound tanker, near head-on to the ego
  - SEA HORIZON:   northbound ferry, crossing from the ego's starboard
  - ANCHOR QUEEN:  anchored bulker, ~stationary

Reports every 10 s per vessel for 30 min with ~10 m position noise and
0.2 kn SOG noise, to exercise the tracker's dead-reckoning realistically.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from ais_replay.geo import LocalFrame, M_PER_DEG_LAT

RNG = np.random.default_rng(42)

LAT0, LON0 = 1.2000, 103.8500
FRAME = LocalFrame(LAT0, LON0)
T0 = datetime(2024, 1, 21, 7, 0, 0, tzinfo=timezone.utc)

# (name, mmsi, start_xy m, sog kn, cog deg, report interval s)
VESSELS = [
    ("LADY MARGAUX", 563012340, (4500.0, 150.0), 11.0, 270.0, 10.0),
    ("SEA HORIZON", 564055670, (2600.0, -1800.0), 14.0, 0.0, 10.0),
    ("ANCHOR QUEEN", 565099880, (1500.0, 800.0), 0.0, 360.0, 10.0),
]

DURATION_S = 1800.0
POS_NOISE_M = 10.0
SOG_NOISE_KN = 0.2


def make_message(name, mmsi, t, lat, lon, sog, cog):
    time_utc = t.strftime("%Y-%m-%d %H:%M:%S.%f000 +0000 UTC")
    return {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": mmsi,
            "ShipName": name,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "time_utc": time_utc,
        },
        "Message": {
            "PositionReport": {
                "UserID": mmsi,
                "Latitude": round(lat, 6),
                "Longitude": round(lon, 6),
                "Sog": round(sog, 1),
                "Cog": round(cog, 1),
                "TrueHeading": int(cog) if cog < 360 else 511,
            }
        },
    }


def main():
    out_path = os.path.join(os.path.dirname(__file__), "sample_strait.jsonl")
    messages = []

    for name, mmsi, (x0, y0), sog_kn, cog_deg, interval in VESSELS:
        v_ms = sog_kn * 0.514444
        psi = np.radians(90.0 - cog_deg)
        for ts in np.arange(0.0, DURATION_S + 1e-9, interval):
            x = x0 + v_ms * np.cos(psi) * ts + RNG.normal(0, POS_NOISE_M)
            y = y0 + v_ms * np.sin(psi) * ts + RNG.normal(0, POS_NOISE_M)
            lat, lon = FRAME.to_latlon(x, y)
            sog = max(sog_kn + RNG.normal(0, SOG_NOISE_KN), 0.0) if sog_kn > 0 else 0.0
            t = T0 + timedelta(seconds=float(ts))
            messages.append((t, make_message(name, mmsi, t, lat, lon, sog, cog_deg)))

    messages.sort(key=lambda m: m[0])
    with open(out_path, "w") as f:
        for _, msg in messages:
            f.write(json.dumps(msg) + "\n")
    print(f"Wrote {len(messages)} messages to {out_path}")


if __name__ == "__main__":
    main()
