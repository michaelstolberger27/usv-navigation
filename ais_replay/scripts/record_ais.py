#!/usr/bin/env python3
"""
Record live AIS traffic from aisstream.io to a JSONL file that
run_replay.py can replay.

Requires a free API key (https://aisstream.io) and `pip install -e .[ais]`.

Usage:
    PYTHONPATH=src:. python3 ais_replay/scripts/record_ais.py \
        --bbox 1.15,103.7,1.35,104.1 \
        --duration 1800 \
        --out singapore_strait.jsonl
    # API key via --api-key or the AISSTREAM_API_KEY env var
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

URL = "wss://stream.aisstream.io/v0/stream"


async def record(api_key, bbox, duration, out_path):
    import websockets

    subscription = json.dumps({
        "APIKey": api_key,
        "BoundingBoxes": [[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]],
        "FilterMessageTypes": ["PositionReport"],
    })
    t_end = time.time() + duration
    count = 0
    with open(out_path, "w") as f:
        async with websockets.connect(URL) as ws:
            await ws.send(subscription)
            while time.time() < t_end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                f.write(raw.decode() if isinstance(raw, bytes) else raw)
                f.write("\n")
                count += 1
                if count % 100 == 0:
                    print(f"{count} messages ({t_end - time.time():.0f} s left)")
    print(f"Recorded {count} messages to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=os.environ.get("AISSTREAM_API_KEY"))
    ap.add_argument("--bbox", required=True,
                    help="lat_min,lon_min,lat_max,lon_max")
    ap.add_argument("--duration", type=float, default=1800.0,
                    help="recording length in seconds")
    ap.add_argument("--out", default="ais_recording.jsonl")
    args = ap.parse_args()

    if not args.api_key:
        ap.error("provide --api-key or set AISSTREAM_API_KEY")
    bbox = [float(p) for p in args.bbox.split(",")]
    if len(bbox) != 4:
        ap.error("--bbox needs lat_min,lon_min,lat_max,lon_max")

    asyncio.run(record(args.api_key, bbox, args.duration, args.out))


if __name__ == "__main__":
    main()
