# Examples

[`realtime_simulation.py`](realtime_simulation.py) runs the COLAV automaton on six
predefined encounter scenarios with a live animated display — no Docker or simulator
required — and saves each run as a GIF:

```bash
pip install -e .[viz]                                 # matplotlib
python realtime_simulation.py --scenario 3            # one scenario
python realtime_simulation.py --all                   # all six
python realtime_simulation.py --scenario 3 --no-unsafe  # hide the unsafe-region overlay
```

GIFs are written to `examples/output/` (gitignored); the curated copies below live in
[`docs/assets/`](../docs/assets/).

Each animation shows the ego trajectory coloured by automaton state (blue S1
waypoint-reaching, red S2 avoidance, orange S3 constant-control), obstacles with their
safety-radius circles, the unsafe-set convex hull, the virtual waypoint V1 while
avoiding, and a live state/time readout.

## Scenario gallery

| **1 — Single stationary obstacle** | **2 — Multiple obstacles (crowded environment)** |
|---|---|
| <img src="../docs/assets/scenario1_single_stationary_obstacle.gif" width="100%"/> | <img src="../docs/assets/scenario2_multiple_obstacles___crowded_environment.gif" width="100%"/> |
| **3 — Head-on encounter** | **4 — Crossing encounter** |
| <img src="../docs/assets/scenario3_head_on_encounter.gif" width="100%"/> | <img src="../docs/assets/scenario4_crossing_encounter.gif" width="100%"/> |
| **5 — Overtaking encounter** | **6 — Multi-vessel crossing** |
| <img src="../docs/assets/scenario5_overtaking_encounter.gif" width="100%"/> | <img src="../docs/assets/scenario6_multi_vessel_crossing.gif" width="100%"/> |
