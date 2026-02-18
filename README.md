# USV Navigation - Collision Avoidance Automaton

A hybrid automaton-based collision avoidance (COLAV) system for Unmanned Surface Vehicles (USVs) that provides provably safe autonomous navigation in dynamic environments.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [Running Examples](#running-examples)
  - [Testing with CommonOcean Simulator](#testing-with-commonocean-simulator)
  - [Batch Evaluation](#batch-evaluation)
  - [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
  - [Automaton Factory](#automaton-factory)
  - [State Dynamics](#state-dynamics)
  - [Guards & Collision Detection](#guards--collision-detection)
  - [Controllers](#controllers)
- [Algorithm References](#algorithm-references)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a **3-state hybrid automaton** that autonomously guides a maritime vessel toward waypoints while dynamically avoiding obstacles. The system uses prescribed-time control theory and unsafe set geometry to guarantee collision-free navigation with formal safety properties.

### Key Features

- **Formal Safety Guarantees**: Uses unsafe set theory with convex hull geometry for provably safe collision avoidance
- **Prescribed-Time Control**: Guaranteed convergence to waypoints within predefined time horizons
- **Multi-Obstacle Support**: Handles multiple static and dynamic obstacles simultaneously
- **Dynamic Obstacle Prediction**: Accounts for obstacle motion and trajectory prediction
- **Real-Time Visualization**: Live animated simulation with state visualization and unsafe set display
- **Modular Architecture**: Clean separation of core automaton logic from CommonOcean simulation integration

## System Architecture

The collision avoidance system operates as a **hybrid automaton** with three states:

```
┌─────────────────────────┐
│  S1: WAYPOINT_REACHING  │  Normal navigation toward target waypoint
│   (Prescribed-Time)     │
└───────────┬─────────────┘
            │ G11 ∧ G12 (Collision threat detected)
            ▼
┌─────────────────────────┐
│ S2: COLLISION_AVOIDANCE │  Navigate to virtual waypoint V1
│  (Unsafe Set + PT)      │  (starboard vertex of unsafe region)
└───────────┬─────────────┘
            │ ¬L1 ∨ ¬L2 (Avoidance complete)
            ▼
┌─────────────────────────┐
│  S3: CONSTANT_CONTROL   │  Hold last command while transitioning
│                         │
└───────────┬─────────────┘
            │ ¬G11 (Line-of-sight clear)
            ▼
        (Back to S1)
```

### State Descriptions

- **S1 (WAYPOINT_REACHING)**: The vessel navigates directly toward its target waypoint using prescribed-time control for guaranteed convergence
- **S2 (COLLISION_AVOIDANCE)**: When obstacles threaten the path, the system computes a virtual waypoint V1 (starboard vertex of the unsafe convex hull) and navigates to it safely
- **S3 (CONSTANT_CONTROL)**: A transition state that maintains the last control command while verifying the avoidance maneuver is complete

### Guard Conditions

- **G11**: Line-of-sight (LOS) to waypoint intersects unsafe regions — cone radius equals `Cs` so any obstacle within the safety radius triggers the guard
- **G12**: At least one obstacle poses collision threat (TCPA ≤ dsafe/v AND DCPA ≤ dsafe)
- **L1**: Vessel has reached virtual waypoint V1 with proper heading alignment
- **L2**: Virtual waypoint V1 is ahead of the vessel (within ±120° of heading)

## Installation

### Prerequisites

- Python 3.10+
- NumPy, Shapely, Matplotlib

### Required Packages

```bash
pip install -e .[viz]
```

### External Dependencies

- `hybrid_automaton`: Hybrid automaton framework with state, transition, decorator support, and async runtime
- `colav_unsafe_set`: Unsafe set computation and obstacle metric calculation (DCPA/TCPA)

### Setup

```bash
git clone <repository-url>
cd usv-navigation
pip install -e .[viz]
```

## Quick Start

### Basic Usage

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

### Running Examples

#### Real-Time Animated Simulation

Run predefined scenarios and save animations as GIFs (no Docker required):

```bash
python examples/realtime_simulation.py                  # Default scenario (1)
python examples/realtime_simulation.py --scenario 3     # Specific scenario
python examples/realtime_simulation.py --all            # Run all scenarios
python examples/realtime_simulation.py --no-unsafe      # Hide unsafe region overlay
```

**Available Scenarios:**
1. Single Stationary Obstacle
2. Multiple Obstacles (Crowded Environment)
3. Head-On Encounter
4. Crossing Encounter
5. Overtaking Encounter
6. Multi-Vessel Crossing

### Testing with CommonOcean Simulator

The COLAV automaton integrates with [commonocean-sim](https://github.com/CommonOcean/commonocean-sim) via the adapter layer in `commonocean_integration/`. A Docker setup provides the full simulation stack (commonocean-sim, Gurobi, VNC) pre-configured.

**Start the container:**

```bash
# Interactive shell
docker/start.sh -it

# Or detached (access via VNC at http://localhost:6080/vnc.html)
docker/start.sh
```

**Run a head-on collision test** (COLAV vessel East-bound vs MPC vessel West-bound):

```bash
cd /app/commonocean-sim/src
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_collision_test.py
```

Saves a trajectory plot and animated GIF to `/app/usv-navigation/output/`.

**Run a single CommonOcean XML scenario:**

```bash
cd /app/commonocean-sim/src
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py <path.xml>
```

The first planning problem is controlled by the COLAV automaton; remaining vessels and dynamic obstacles are handled by commonocean-sim defaults.

> **Note:** Traffic trajectories are automatically interpolated from the scenario's 10s timestep to the simulation's 1s timestep so both vessels use the same physical time rate.

### Batch Evaluation

Evaluate the COLAV automaton across a large set of CommonOcean XML scenarios:

```bash
cd /app/commonocean-sim/src

# Run all scenarios
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py

# Quick test with a small subset
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py --limit 10

# Resume a previous run (skips already-completed scenarios)
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py --resume

# Custom scenarios directory and output
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py \
    --scenarios-dir /app/scenarios \
    --output-dir /app/usv-navigation/output/batch_eval \
    --limit 50 --start 0
```

Results are saved incrementally to `output/batch_eval/results.csv` with per-scenario metrics including CPA distance, goal reached, collision detected, encounter type, and automaton state time distribution. Summary plots are generated on completion.

### Visualization

The simulation generates trajectory plots and animations showing:

- **Vessel trajectory** with state-based colouring (Blue: S1, Red: S2, Orange: S3)
- **Obstacle positions** with safety radius circles
- **Unsafe set regions** (convex hulls)
- **Virtual waypoint V1** (when in avoidance mode)
- **LOS cone** and heading arrows
- **Current state indicator** (S1/S2/S3) with time readout

## Project Structure

```
usv-navigation/
├── src/
│   └── colav_automaton/               # Core automaton — no CommonOcean dependencies
│       ├── __init__.py                # Package exports (ColavAutomaton, HeadingControlProvider)
│       ├── automaton.py               # Automaton factory
│       ├── controllers/
│       │   ├── prescribed_time.py     # Prescribed-time heading controller & HeadingControlProvider
│       │   ├── virtual_waypoint.py    # Virtual waypoint V1 computation (COLREGs-compliant)
│       │   └── unsafe_sets.py         # Unsafe set geometry, LOS cone, collision threat detection
│       ├── dynamics/
│       │   └── dynamics.py            # State flow dynamics (S1/S2 navigation, S3 constant)
│       ├── guards/
│       │   ├── guards.py              # Transition guards (G11∧G12, ¬L1∨¬L2, ¬G11)
│       │   └── conditions.py          # Collision detection logic (G11, G12, L1, L2)
│       ├── resets/
│       │   └── resets.py              # State reset handlers (V1 computation & waypoint stack)
│       └── invariants/
│           └── invariants.py          # State invariant conditions
├── commonocean_integration/           # CommonOcean-specific code (requires Docker)
│   ├── sim_utils.py                   # Shared utilities: trajectory interpolation, config loading
│   ├── adapters/
│   │   ├── controller.py              # HybridAutomatonController (commonocean-sim adapter)
│   │   └── vessel_factory.py          # ColavVesselFactory (creates YP-model vessels)
│   ├── evaluation/
│   │   └── metrics.py                 # Per-scenario metrics: CPA, goal reached, encounter type
│   └── scripts/
│       ├── batch_evaluate.py          # Batch runner across many XML scenarios
│       ├── commonocean_scenario.py    # Single scenario runner with pyglet display
│       ├── commonocean_collision_test.py  # Head-on collision test + GIF output
│       └── plot_trajectories.py       # Trajectory plot generator for selected scenarios
├── examples/
│   └── realtime_simulation.py         # Standalone animated simulation (no Docker required)
├── docker/
│   ├── Dockerfile                     # Full simulation stack (commonocean-sim + Gurobi + VNC)
│   ├── docker-compose.yml             # Service definition with volume mounts
│   └── start.sh                       # Helper to start/stop the container
├── pyproject.toml
└── README.md
```

## Key Components

### Automaton Factory

The [`ColavAutomaton()`](src/colav_automaton/automaton.py) function creates a configured hybrid automaton:

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
    v1_buffer: float = 0.0      # Virtual waypoint clearance buffer (meters)
)
```

> **Note:** `delta` and `dsafe` are derived automatically:
> - `delta = max(5.0, v * tp * 0.5)` — arrival tolerance
> - `dsafe = Cs + (v * 2) * tp` — safe distance threshold

**CommonOcean evaluation uses:** `Cs=300.0`, `tp=3.0`, `a=1.67`, `eta=3.5` (real-world scale, dt=1s).

**Stability condition:** The prescribed-time controller requires `a * dt < 2` and `tp > dt`. With `a=1.67` and `dt=1.0`: `a*dt = 1.67 < 2` ✓

### State Dynamics

Two continuous dynamics functions in [`dynamics.py`](src/colav_automaton/dynamics/dynamics.py):

- **`waypoint_navigation_dynamics()`**: Shared by S1 and S2 — uses prescribed-time control to navigate toward the current waypoint (goal in S1, virtual waypoint V1 in S2)
- **`constant_control_dynamics()`**: Used by S3 — maintains current heading (zero yaw rate)

### Guards & Collision Detection

Transition logic in [`guards.py`](src/colav_automaton/guards/guards.py) and [`conditions.py`](src/colav_automaton/guards/conditions.py):

**G11 (LOS Intersection Check):**
- Creates a cone from vessel position toward waypoint with radius `Cs` (= safety radius)
- Tests intersection with unsafe set polygon using Shapely
- Any obstacle within `Cs` of the direct path triggers avoidance

**G12 (Threat Assessment):**
- For each obstacle: `if (TCPA ≤ dsafe/v) AND (DCPA ≤ dsafe): threat = True`
- Uses `check_collision_threat()` from `controllers.unsafe_sets`

**Post-avoidance waypoint recovery:**
- On S2/S3 → S1 transition, waypoints that are now behind the vessel are skipped automatically, preventing backtracking after an avoidance manoeuvre

### Controllers

#### Prescribed-Time Controller ([`prescribed_time.py`](src/colav_automaton/controllers/prescribed_time.py))

- **`compute_prescribed_time_control()`**: Computes the heading control law guaranteeing convergence to line-of-sight within time `tp`
- **`HeadingControlProvider`**: Async control provider that runs alongside the automaton, computing control input `u` each cycle

#### Virtual Waypoint ([`virtual_waypoint.py`](src/colav_automaton/controllers/virtual_waypoint.py))

- **`compute_v1()`**: Selects the starboard-most unsafe set vertex ahead of the vessel (within ±90° of heading), with optional buffer. COLREGs-compliant starboard preference.

#### Unsafe Set Geometry ([`unsafe_sets.py`](src/colav_automaton/controllers/unsafe_sets.py))

- **`get_unsafe_set_vertices()`**: Convex hull vertices of unsafe regions (with swept region support for moving obstacles)
- **`create_los_cone()`**: Convex cone from vessel toward waypoint for G11 intersection test
- **`check_collision_threat()`**: DCPA/TCPA-based threat assessment

## Algorithm References

- **Prescribed-Time Control**: Guaranteed convergence within predefined time horizon
- **Unsafe Set Theory**: Convex hull representation of collision regions
- **Hybrid Automaton Framework**: Formal modelling of discrete state transitions with continuous dynamics
- **COLREGS Compliance**: Starboard avoidance manoeuvres following maritime collision regulations

## Acknowledgments

- `hybrid_automaton`: Hybrid system modelling framework with async runtime
- `colav_unsafe_set`: Unsafe set computation and obstacle metric calculation (DCPA/TCPA)
- [commonocean-sim](https://github.com/CommonOcean/commonocean-sim): Maritime scenario simulator used for evaluation
