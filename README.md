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

- `hybrid_automaton`: Hybrid automaton framework with state, transition, and decorator support
- `hybrid_automaton_runner`: Simulation orchestration and data collection
- `colav_controllers`: Prescribed-time controllers, unsafe set geometry, LOS cone, V1 computation, and collision threat detection

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
from colav_automaton import ColavAutomaton
from hybrid_automaton_runner import AutomatonRunner

async def main():
    # Create automaton with single obstacle
    automaton = ColavAutomaton(
        waypoint_x=20.0,
        waypoint_y=20.0,
        obstacles=[(10.0, 10.0, 0.0, 0.0)],  # (x, y, velocity, heading)
        Cs=2.0,   # Safety radius (meters)
        v=12.0,   # Vessel velocity (m/s)
        tp=1.0    # Prescribed time (seconds)
    )

    # Run simulation
    runner = AutomatonRunner(automaton, sampling_rate=0.001)
    await runner.run(
        x0=np.array([0.0, 0.0, 0.0]),  # [x, y, heading]
        duration=30.0,
        dt=0.1,
        integrate=True,
        real_time_mode=False
    )

    results = runner.get_results()

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
│   ├── main.py                      # Basic example with plotting
│   └── colav_automaton/
│       ├── __init__.py              # Package exports & version
│       ├── automaton.py             # Main automaton factory
│       ├── dynamics/
│       │   └── dynamics.py          # State flow dynamics (shared S1/S2, S3)
│       ├── guards/
│       │   ├── guards.py            # Transition guards (G11∧G12, ¬L1∨¬L2, ¬G11)
│       │   └── conditions.py        # Collision detection logic (G11, G12, L1, L2)
│       ├── resets/
│       │   └── resets.py            # State reset handlers (V1 computation & stack mgmt)
│       └── invariants/
│           └── invariants.py        # State invariant conditions
├── examples/
│   ├── realtime_simulation.py       # Real-time animated simulation with GIF export
│   └── output/                      # Generated GIF animations
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

Collision detection and transition logic using functions from `colav_controllers`:

- **Line-of-Sight (LOS) Cone**: Checks if path intersects unsafe regions via `create_los_cone()` and `compute_unified_unsafe_region()`
- **TCPA/DCPA Metrics**: Time and distance to closest point of approach via `check_collision_threat()`
- **Heading Alignment**: Ensures proper orientation before state transitions (~3° threshold)
- **V1 Ahead Check**: Verifies virtual waypoint is within ±120° of heading before transitioning

### Unsafe Set Geometry (from `colav_controllers`)

The geometric collision detection functions are provided by the external `colav_controllers` package:

- **`get_unsafe_set_vertices()`**: Generates convex hull of buffered obstacle regions
- **`create_los_cone()`**: Forms cone from vessel position toward waypoint
- **`compute_unified_unsafe_region()`**: Aggregates multi-obstacle unsafe sets
- **`compute_v1()`**: Computes virtual waypoint V1 (starboard vertex of unsafe hull)
- **`check_collision_threat()`**: DCPA/TCPA-based threat assessment

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
- Uses `check_collision_threat()` from `colav_controllers`
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
automaton = ColavAutomaton(
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
automaton = ColavAutomaton(
    waypoint_x=50.0,
    waypoint_y=0.0,
    obstacles=[(30.0, 0.0, 3.0, 3.14159)],  # Moving toward vessel
    Cs=3.0,
    v=5.0
)
```

### Tight Maneuvering

```python
automaton = ColavAutomaton(
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

### Running Simulations

```python
from hybrid_automaton_runner import AutomatonRunner
import numpy as np

runner = AutomatonRunner(automaton, sampling_rate=0.001)
await runner.run(
    x0=np.array([x0, y0, psi0]),
    duration=30.0,
    dt=0.1,
    integrate=True,
    real_time_mode=False
)

results = runner.get_results()
```

Returns simulation results including continuous states, automaton states, and transition times.

## Algorithm References

This implementation is based on:

- **Prescribed-Time Control**: Guaranteed convergence within predefined time horizon
- **Unsafe Set Theory**: Convex hull representation of collision regions
- **Hybrid Automaton Framework**: Formal modeling of discrete state transitions with continuous dynamics
- **COLREGS Compliance**: Starboard avoidance maneuvers following maritime collision regulations

## Acknowledgments

This project uses the following frameworks:

- `hybrid_automaton`: Hybrid system modeling framework
- `hybrid_automaton_runner`: Simulation orchestration and data collection
- `colav_controllers`: Maritime collision avoidance controllers and geometric collision detection
