# USV Navigation - Collision Avoidance Automaton

A hybrid automaton-based collision avoidance (COLAV) system for Unmanned Surface Vehicles (USVs) that provides provably safe autonomous navigation in dynamic environments.

## Overview

This project implements a **3-state hybrid automaton** that autonomously guides a maritime vessel toward waypoints while dynamically avoiding obstacles. The system uses prescribed-time control theory and unsafe set geometry to guarantee collision-free navigation with formal safety properties.

### Key Features

- **Formal Safety Guarantees**: Uses unsafe set theory with convex hull geometry for provably safe collision avoidance
- **Prescribed-Time Control**: Guaranteed convergence to waypoints within predefined time horizons
- **Multi-Obstacle Support**: Handles multiple static and dynamic obstacles simultaneously
- **Dynamic Obstacle Prediction**: Accounts for obstacle motion and trajectory prediction
- **Real-Time Visualization**: Live animated simulation with state visualization and unsafe set display
- **Modular Architecture**: Clean separation of dynamics, guards, resets, and geometric computations

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

- **G11**: Line-of-sight (LOS) to waypoint intersects unsafe regions (collision cone check)
- **G12**: At least one obstacle poses collision threat (TCPA ≤ dsafe/v AND DCPA ≤ dsafe)
- **L1**: Vessel has reached virtual waypoint V1 with proper heading alignment
- **L2**: Virtual waypoint V1 is ahead of the vessel (within ±120° of heading)

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- Shapely

### Required Packages

```bash
pip install numpy matplotlib shapely
```

### External Dependencies

This system relies on the following custom packages (ensure they are installed or available in your Python path):

- `hybrid_automaton`: Hybrid automaton framework with state, transition, decorator support, and async runtime
- `colav_unsafe_set`: Unsafe set computation and obstacle metric calculation (DCPA/TCPA)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd usv-navigation

# Ensure src/ is on your Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
```

## Quick Start

### Basic Usage

```python
import asyncio
import numpy as np
from colav_automaton import ColavAutomaton, HeadingControlProvider
from hybrid_automaton import Automaton, RunResult

async def main():
    # Create automaton with single obstacle
    ha: Automaton = ColavAutomaton(
        waypoint_x=10.0,
        waypoint_y=9.0,
        obstacles=[(5.0, 4.5, 0.0, 0.0)],  # (x, y, velocity, heading)
        Cs=2.0,   # Safety radius (meters)
        v=12.0,   # Vessel velocity (m/s)
        tp=1.0    # Prescribed time (seconds)
    )

    # Controller runs asynchronously to the automaton
    controller = HeadingControlProvider(ha)

    # Run simulation
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

Run predefined scenarios and save animations as GIFs:

```bash
cd examples
python realtime_simulation.py                      # Default scenario (1)
python realtime_simulation.py --scenario 3         # Specific scenario
python realtime_simulation.py --all                # Run all scenarios
python realtime_simulation.py --no-los             # Hide LOS cone overlay
python realtime_simulation.py --no-unsafe          # Hide unsafe region overlay
```

**Available Scenarios:**
1. Single Stationary Obstacle
2. Multiple Obstacles (Crowded Environment)
3. Head-On Encounter
4. Crossing Encounter
5. Overtaking Encounter
6. Multi-Vessel Crossing

## Project Structure

```
usv-navigation/
├── src/
│   ├── main.py                        # Basic example with plotting
│   └── colav_automaton/
│       ├── __init__.py                # Package exports (ColavAutomaton, HeadingControlProvider)
│       ├── automaton.py               # Main automaton factory
│       ├── controllers/
│       │   ├── __init__.py            # Controller module exports
│       │   ├── prescribed_time.py     # Prescribed-time heading controller & HeadingControlProvider
│       │   ├── virtual_waypoint.py    # Virtual waypoint V1 computation (COLREGs-compliant)
│       │   └── unsafe_sets.py         # Unsafe set geometry, LOS cone, collision threat detection
│       ├── dynamics/
│       │   ├── __init__.py            # Dynamics module exports
│       │   └── dynamics.py            # State flow dynamics (shared S1/S2, S3)
│       ├── guards/
│       │   ├── __init__.py            # Guards module exports
│       │   ├── guards.py             # Transition guards (G11∧G12, ¬L1∨¬L2, ¬G11)
│       │   └── conditions.py          # Collision detection logic (G11, G12, L1, L2)
│       ├── resets/
│       │   ├── __init__.py            # Resets module exports
│       │   └── resets.py              # State reset handlers (V1 computation & stack mgmt)
│       └── invariants/
│           ├── __init__.py            # Invariants module exports
│           └── invariants.py          # State invariant conditions
├── examples/
│   ├── realtime_simulation.py         # Real-time animated simulation with GIF export
│   └── output/                        # Generated GIF animations
└── README.md
```

## Key Components

### Automaton Factory ([automaton.py](src/colav_automaton/automaton.py))

The `ColavAutomaton()` function creates a configured hybrid automaton with the following parameters:

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
    v1_buffer: float = 0.0      # Virtual waypoint offset (meters)
)
```

> **Note:** `delta` and `dsafe` are derived automatically:
> - `delta = max(5.0, v * tp * 0.5)` — arrival tolerance
> - `dsafe = Cs + (v * 2) * tp` — safe distance threshold

### State Dynamics ([dynamics/dynamics.py](src/colav_automaton/dynamics/dynamics.py))

Two continuous dynamics functions define vessel motion:

- **`waypoint_navigation_dynamics()`**: Shared by S1 and S2 — uses prescribed-time control to navigate toward the top of the waypoints stack (goal waypoint in S1, virtual waypoint V1 in S2)
- **`constant_control_dynamics()`**: Used by S3 — maintains current heading with zero turning rate (straight-line motion)

### Guard Conditions ([guards/guards.py](src/colav_automaton/guards/guards.py), [guards/conditions.py](src/colav_automaton/guards/conditions.py))

Collision detection and transition logic using functions from the [controllers](src/colav_automaton/controllers/) module:

- **Line-of-Sight (LOS) Cone**: Checks if path intersects unsafe regions via `create_los_cone()` and `compute_unified_unsafe_region()`
- **TCPA/DCPA Metrics**: Time and distance to closest point of approach via `check_collision_threat()`
- **Heading Alignment**: Ensures proper orientation before state transitions (~3° threshold)
- **V1 Ahead Check**: Verifies virtual waypoint is within ±120° of heading before transitioning

### Controllers ([controllers/](src/colav_automaton/controllers/))

#### Prescribed-Time Controller ([prescribed_time.py](src/colav_automaton/controllers/prescribed_time.py))

- **`compute_prescribed_time_control()`**: Computes the heading control law that guarantees convergence to line-of-sight within time `tp`. Uses feedforward (LOS rate) + prescribed-time feedback with singularity avoidance.
- **`HeadingControlProvider`**: Async control provider class that runs alongside the automaton evaluation loop, reading the current continuous state and computing the control input `u` each cycle.

#### Virtual Waypoint ([virtual_waypoint.py](src/colav_automaton/controllers/virtual_waypoint.py))

- **`compute_v1()`**: Computes virtual waypoint V1 — the starboard-most unsafe set vertex ahead of the ship (within ±90° of heading), with optional buffer for extra clearance. COLREGs-compliant starboard preference.
- **`default_vertex_provider()`**: Creates 8 vertices around each obstacle at distance `Cs` for circular obstacle approximation.

#### Unsafe Set Geometry ([unsafe_sets.py](src/colav_automaton/controllers/unsafe_sets.py))

Wraps the external `colav_unsafe_set` package and provides geometric collision detection:

- **`get_unsafe_set_vertices()`**: Generates convex hull vertices of unsafe regions, with support for swept regions (predicted obstacle trajectories)
- **`create_los_cone()`**: Forms a convex cone from vessel position toward waypoint
- **`compute_unified_unsafe_region()`**: Aggregates multi-obstacle unsafe sets with swept region support
- **`check_collision_threat()`**: DCPA/TCPA-based threat assessment using `colav_unsafe_set` metrics

## Technical Details

### Continuous State Vector

```python
x = [x, y, psi]
```

- `x, y`: Position in meters (East-North coordinate frame)
- `psi`: Heading in radians, normalized to [-π, π]

### Collision Detection Algorithms

**G11 (LOS Intersection Check):**
- Creates cone with radius `v * min(tp, 1.0)` around vessel
- Tests intersection with unsafe set polygon using Shapely
- Caps cone at 1 second to prevent false positives for passed obstacles

**G12 (Threat Assessment):**
- For each obstacle: `if (TCPA ≤ dsafe/v) AND (DCPA ≤ dsafe): threat = True`
- Uses `check_collision_threat()` from `controllers.unsafe_sets`
- Where `dsafe = Cs + (v * 2) * tp` provides sufficient lead time for maneuvering

**Virtual Waypoint V1:**
- Computed as starboard vertex of unsafe convex hull
- Accounts for swept region when obstacles are moving
- Buffer distance `v1_buffer` adds extra clearance

## Visualization

### Trajectory Plots

The simulation generates plots showing:

- **Vessel trajectory** with state-based coloring:
  - Blue: S1 (Waypoint Reaching)
  - Red: S2 (Collision Avoidance)
  - Orange: S3 (Constant Control)
- **Obstacle positions** with safety radius circles
- **Unsafe set regions** (convex hulls)
- **Virtual waypoint V1** (when in avoidance mode)
- **Start/End markers**

### Real-Time Animation

Live visualization displays:

- Agent vessel with heading arrow
- Dynamic obstacle motion
- LOS cone (cyan, toggleable)
- Unsafe regions (red, toggleable)
- Virtual waypoint V1 (magenta diamond)
- Current state indicator (S1/S2/S3)
- Time and position readout

## Configuration Examples

### Multiple Dynamic Obstacles

```python
ha = ColavAutomaton(
    waypoint_x=30.0,
    waypoint_y=30.0,
    obstacles=[
        (15.0, 10.0, 2.0, 1.57),   # (x, y, velocity, heading)
        (20.0, 20.0, 1.5, 3.14),
        (25.0, 15.0, 1.0, 0.0)
    ],
    Cs=2.5,
    v=12.0,
    tp=1.0
)
```

### Head-On Encounter

```python
ha = ColavAutomaton(
    waypoint_x=100.0,
    waypoint_y=0.0,
    obstacles=[(70.0, 0.0, 2.0, np.pi)],  # Moving toward vessel
    Cs=5.0,
    v=12.0,
    tp=1.0,
    v1_buffer=3.0
)
```

### Tight Maneuvering

```python
ha = ColavAutomaton(
    waypoint_x=20.0,
    waypoint_y=20.0,
    obstacles=[(10.0, 10.0, 0.0, 0.0)],
    Cs=1.5,           # Smaller safety radius
    v=8.0,            # Slower vessel speed
    tp=0.5,           # Faster response time
    v1_buffer=2.0     # Extra clearance
)
```

## API Reference

### Main Factory

```python
from colav_automaton import ColavAutomaton

automaton = ColavAutomaton(**config)
```

Returns a configured `Automaton` object ready for simulation.

### HeadingControlProvider

```python
from colav_automaton import HeadingControlProvider

controller = HeadingControlProvider(automaton)
```

Async control provider that computes the prescribed-time heading control input `u` each cycle. Pass as the `control_states_provider` argument to `automaton.activate()`.

### Running Simulations

```python
from colav_automaton import ColavAutomaton, HeadingControlProvider
from hybrid_automaton import RunResult
import numpy as np

ha = ColavAutomaton(**config)
controller = HeadingControlProvider(ha)

results: RunResult = await ha.activate(
    initial_continuous_state=np.array([x0, y0, psi0]),
    initial_control_input_states={'u': np.array([0.0])},
    enable_real_time_mode=False,
    enable_self_integration=True,
    delta_time=0.1,
    timeout_sec=30.0,
    continuous_state_sampler_enabled=True,
    continuous_state_sampler_rate=100,
    control_states_provider=controller,
    control_states_provision_rate=100,
)
```

Returns `RunResult` containing simulation data. When `continuous_state_sampler_enabled=True` and an `output_dir` is provided, state trajectories and transition logs are written to CSV and log files.

## Algorithm References

This implementation is based on:

- **Prescribed-Time Control**: Guaranteed convergence within predefined time horizon
- **Unsafe Set Theory**: Convex hull representation of collision regions
- **Hybrid Automaton Framework**: Formal modeling of discrete state transitions with continuous dynamics
- **COLREGS Compliance**: Starboard avoidance maneuvers following maritime collision regulations

## Acknowledgments

This project uses the following frameworks:

- `hybrid_automaton`: Hybrid system modeling framework with async runtime
- `colav_unsafe_set`: Unsafe set computation and obstacle metric calculation (DCPA/TCPA)
