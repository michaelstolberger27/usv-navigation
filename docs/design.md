# Design reference

Component-level reference for the core automaton in
[`src/colav_automaton/`](../src/colav_automaton/). For the architecture overview,
state machine diagram, and evaluation results, see the [root README](../README.md).

## Contents

- [Automaton factory and parameters](#automaton-factory-and-parameters)
- [Two runtimes, one automaton](#two-runtimes-one-automaton)
- [State dynamics](#state-dynamics)
- [Guards in detail](#guards-in-detail)
- [Controllers](#controllers)
- [Algorithm references](#algorithm-references)

## Automaton factory and parameters

The [`ColavAutomaton()`](../src/colav_automaton/automaton.py) factory creates a
configured hybrid automaton:

```python
ColavAutomaton(
    waypoint_x: float,           # Target x-coordinate
    waypoint_y: float,           # Target y-coordinate
    obstacles: list,             # [(x, y, velocity, heading), ...]
    Cs: float = 2.0,            # Safety radius (meters)
    v: float = 12.0,            # Vessel velocity (m/s)
    a: float = 1.67,            # System dynamics parameter
    eta: float = 3.5,           # Controller gain
    tp: float = 1.0,            # Prescribed time (seconds)
    v1_buffer: float = 0.0,     # Virtual waypoint clearance buffer (meters)
    K: float = 0.35,            # Risk-index threshold to enter avoidance (G22)
    K_off: float = 0.25,        # Risk-index threshold to resume from S3 (hysteresis, < K)
    m: float = 3.0,             # Input constraint bound |u| <= m (paper eq 9)
    # ...plus six {dcpa,tcpa,dist}_beta{1,2} overrides for the G22 risk-index bounds
)
```

Derived automatically:

- `delta = max(5.0, v * tp * 0.5)` — arrival tolerance
- `dsafe = Cs + v * tp` — safe distance threshold (paper eq 14)

**Stability condition:** the prescribed-time controller requires `a * dt < 2` and
`tp > dt`. With `a=1.67` and `dt=1.0`: `a*dt = 1.67 < 2` ✓

## Two runtimes, one automaton

The guards, dynamics, and resets are plain functions shared by two executives.

### Deterministic synchronous runtime (preferred for new integrations)

[`SyncColavRuntime`](../src/colav_automaton/sync_runtime.py) steps the automaton
tick-by-tick in sim time — identical inputs give bit-identical trajectories. It takes
the same parameters as the factory (via `waypoint=(x, y)`, `initial_state`, plus one
addition: `tp_control`, an optional override for the prescribed-time horizon used by
the control law, defaulting to `tp` — this lets an integration use a longer control
horizon than the geometric `tp` that sizes `dsafe`). Two stepping modes:

- **`step(dt, obstacles)`** — the runtime integrates the vessel itself (used by the
  [AIS replay runner](../ais_traffic/README.md) and the standalone examples).
- **`step_external(...)`** — the caller owns vessel integration and feeds pose back;
  the runtime only evaluates guards and returns mode + control (used by the
  [CommonOcean adapter](../commonocean_integration/README.md) and the
  [ROS 2 nodes](../ros2/README.md)).

### Async wall-clock runtime

`ColavAutomaton` runs on the `hybrid-automaton` framework's async runtime with
background state sampling and a control provider:

```python
import asyncio
import numpy as np
from colav_automaton import ColavAutomaton, HeadingControlProvider
from hybrid_automaton import Automaton, RunResult

async def main():
    ha: Automaton = ColavAutomaton(
        waypoint_x=10.0,
        waypoint_y=9.0,
        obstacles=[(5.0, 4.5, 0.0, 0.0)],  # (x, y, velocity, heading)
        Cs=2.0,   # Safety radius (meters)
        v=12.0,   # Vessel velocity (m/s)
        tp=1.0    # Prescribed time (seconds)
    )

    controller = HeadingControlProvider(ha)

    results: RunResult = await ha.activate(
        initial_continuous_state=np.array([0.0, 0.0, 0.0]),  # [x, y, heading]
        initial_control_input_states={'u': np.array([0.0])},
        enable_real_time_mode=False,
        enable_self_integration=True,
        delta_time=0.1,
        timeout_sec=15.0,
        continuous_state_sampler_enabled=True,
        continuous_state_sampler_rate=100,
        control_states_provider=controller,
        control_states_provision_rate=100,
    )

    print(results)

asyncio.run(main())
```

The async runtime is wall-clock driven, so its results vary run to run; the
evaluation figures in the root README come from the synchronous runtime.

## State dynamics

Two continuous dynamics functions in
[`dynamics.py`](../src/colav_automaton/dynamics/dynamics.py):

- **`waypoint_navigation_dynamics()`** — shared by S1 and S2: prescribed-time control
  toward the current waypoint (the goal in S1, the virtual waypoint V1 in S2)
- **`constant_control_dynamics()`** — S3: maintains current heading (zero yaw rate)

## Guards in detail

Transition logic lives in [`guards.py`](../src/colav_automaton/guards/guards.py)
(the composite transition guards) and
[`conditions.py`](../src/colav_automaton/guards/conditions.py) (the individual
conditions G11, G22, G23, L1, L2, and the risk index).

### G11 — LOS intersection check

- Builds a cone from the vessel position toward the waypoint with radius `Cs`
  (via `effective_tp = Cs/v`), so any obstacle within the safety radius of the
  direct path triggers the guard
- Tests intersection against the unified unsafe region using Shapely
- The region is **motion-aware**: obstacles carry their real velocity, so it includes
  TCPA-predicted positions — crossing traffic triggers avoidance before it reaches
  the path

### G22 — risk index

```
RI(DCPA, TCPA, d_s) = ⅓ · (F(DCPA) + F(TCPA) + F(d_s)) ≥ K
```

with the paper's piecewise quadratic `F(z)` mapping each metric onto [0, 1]
(`F = 1` below `beta1`, `F = 0` above `beta2`, two quadratic arcs joined at the
midpoint in between). The risk index is the maximum over all *approaching* obstacles,
so it triggers avoidance much earlier than a plain distance check for converging
traffic. Default bounds (scaled to meters/seconds from the paper's nmi/min values):

| Metric | beta1 | beta2 |
|---|---|---|
| DCPA | 463 m | 926 m |
| TCPA | 120 s | 240 s |
| Distance | 148 m | 463 m |

All six are overridable via the factory/runtime `{dcpa,tcpa,dist}_beta{1,2}` kwargs.

### L1 / L2 — avoidance-complete checks

- **L1**: the vessel has not yet reached V1 (pure distance check against `delta`)
- **L2**: V1 is still ahead of the vessel (within ±90° of heading)

S2 exits to S3 when either fails (`¬L1 ∨ ¬L2`): the manoeuvre point has been reached
or has fallen behind.

### G23 + K_off — resume check with hysteresis

- The ship leaves S3 only when the LOS to the waypoint is clear of the unsafe region
  **and** the risk index has dropped below `K_off < K`
- The resume check deliberately uses **static-only** unsafe regions (current obstacle
  positions, no TCPA prediction) — the predictive region that makes G11 trigger early
  is too conservative as a resume condition and would delay resumption indefinitely
- The asymmetric thresholds (enter at `K ≥ 0.35`, resume below `K_off = 0.25`) prevent
  rapid avoid/resume cycling against still-converging traffic; before the hysteresis
  existed, that cycling produced both collisions and non-reproducible outcomes

## Controllers

### Prescribed-time controller ([`prescribed_time.py`](../src/colav_automaton/controllers/prescribed_time.py))

- **`compute_prescribed_time_control()`** — the heading control law guaranteeing
  convergence to the line of sight within time `tp`
- **`HeadingControlProvider`** — async control provider that runs alongside the async
  runtime, computing the control input `u` each cycle (the sync runtime calls the
  control law directly)

### Virtual waypoint ([`virtual_waypoint.py`](../src/colav_automaton/controllers/virtual_waypoint.py))

- **`compute_v1()`** — evaluates both the starboard-most and port-most unsafe-set
  vertices ahead of the vessel (within ±90° of heading), predicts the CPA each would
  yield, and picks starboard unless port is >10% better (COLREGs starboard preference
  with a Rule 17b port escape), with optional buffer
- V1 candidates come from a **swept** unsafe hull: each obstacle's predicted trajectory
  is sampled over a horizon capped at `max(60 s, 3·dsafe/v)` — the cap keeps V1 at a
  reachable distance for slow obstacles, where sweeping over the full TCPA would place
  it unreachably far

### Unsafe set geometry ([`unsafe_sets.py`](../src/colav_automaton/controllers/unsafe_sets.py))

Two distinct entry points into the `colav_unsafe_set` dependency — used for different
jobs:

- **`get_unsafe_set_vertices()`** — convex hull **vertices** (with swept-region
  support); feeds V1 selection
- **`compute_unified_unsafe_region()`** — a Shapely **polygon** for guard intersection
  tests; motion-aware for G11, `static_only` for the G23 resume check

Both return `None` on degenerate geometry (the upstream library throws), and callers
carry explicit fallback paths for that case.

## Algorithm references

- **Prescribed-Time Control**: guaranteed convergence within a predefined time horizon
- **Unsafe Set Theory**: convex hull representation of collision regions
- **Hybrid Automaton Framework**: formal modelling of discrete state transitions with
  continuous dynamics
- **COLREGS Compliance**: starboard avoidance manoeuvres following maritime collision
  regulations
