# ROS 2 integration (`colav_ros`)

A ROS 2 node that wraps the COLAV hybrid automaton, plus two worlds to
run it against: a closed-loop fake-world node (no simulator required)
and a headless Gazebo world driven through `ros_gz_bridge`. The verified
C++ port (`colav_cpp`) speaks the same topics as the Python node.

Tested on **ROS 2 Jazzy**.

## Packages

| Package | Build type | Contents |
|---|---|---|
| `colav_interfaces` | `ament_cmake` | `Obstacle.msg`, `ObstacleArray.msg` |
| `colav_ros` | `ament_python` | `colav_node` (Python controller), `fake_world` (stand-in plant + traffic), `gz_obstacles` (Gazebo obstacle adapter), `demo.launch.py`, `gazebo_demo.launch.py`, `worlds/colav_demo.sdf`, a `launch_testing` smoke test |
| `colav_cpp` | `ament_cmake` | `colav_core` — a C++ reimplementation of the controller verified bit-identical to the Python core — plus `colav_node_cpp`, the C++ rclcpp node that links it |

## C++ controller (`colav_cpp`)

`colav_core` is a from-scratch C++ port of the deterministic controller
(`SyncColavRuntime`): the prescribed-time control law, the risk index / G22,
the geometry guards G11/G23 (its own convex hull + separating-axis polygon
intersection, replacing scipy + shapely), V1 selection, and the step/
transition runtime. It depends on no ROS or simulator libraries.

Python (`colav_automaton`) stays the source of truth; the C++ is verified
against it. `scripts/gen_reference.py` emits reference vectors and five
gtests cross-check each layer:

| Layer | Result vs Python |
|---|---|
| Prescribed-time control + L1/L2 | `u` bit-exact 1000/1000 |
| Risk index / G22 | `ri` ≤1 ULP; G22 decision identical 1000/1000 |
| Geometry guards G11/G23 | decisions match 1500/1500 |
| V1 selection | point match 1499/1500 (1 boundary case) |
| **Full trajectory (842-step head-on)** | **bit-identical: 0 position/heading diff, modes & transitions 842/842** |

`colav_node_cpp` is then pure ROS plumbing linking `colav_core` — a drop-in
replacement for the Python node speaking the same topics:

```bash
cd ros2 && colcon build && source install/setup.bash
export COLAV_REPO=$(cd .. && pwd)
ros2 run colav_ros fake_world &        # Python world
ros2 run colav_cpp colav_node_cpp      # C++ controller
colcon test   # C++ cross-checks + launch_testing smoke test of the Python pair
```

## Design

Asynchronous I/O at the edges, deterministic control in the middle — the
standard time-triggered pattern:

- subscriptions cache the latest ego pose and obstacle list as messages arrive;
- a fixed-rate timer reads that snapshot and steps `SyncColavRuntime.step_external`
  once per tick (the same deterministic core the CommonOcean adapter and the
  AIS replay runner drive);
- the node publishes a `Twist` command and never integrates the vessel — the
  world on the other side (fake_world or the bundled Gazebo world here, real
  hardware later) owns integration and feeds pose back.

```
            ego_odom (nav_msgs/Odometry)
 fake_world ───────────────────────────▶ colav_node
   (plant   ◀───────────────────────────   (SyncColavRuntime)
  + traffic)        cmd (geometry_msgs/Twist)
            obstacles (colav_interfaces/ObstacleArray) ▶
```

Because the controller depends only on the topic contract, swapping
`fake_world` for Gazebo is a launch-file change, not a code change —
`gazebo_demo.launch.py` is exactly that: the same controller binary
remapped onto bridged Gazebo topics.

## Build & run

The node imports the simulator-independent core from this repo's `src/`.
Point `COLAV_REPO` at the repo root so it is importable (or `pip install -e .`
the core into the ROS Python environment):

```bash
source /opt/ros/jazzy/setup.bash
export COLAV_REPO=$(cd .. && pwd)          # repo root (has src/colav_automaton)

cd ros2
colcon build
source install/setup.bash

ros2 launch colav_ros demo.launch.py
```

Expected: the ego enters `COLLISION_AVOIDANCE` as the head-on obstacle
approaches, holds in `CONSTANT_CONTROL`, resumes `WAYPOINT_REACHING`, and
logs `goal reached`. Inspect the live graph with:

```bash
ros2 topic echo /colav_state      # S1/S2/S3 label each tick
ros2 topic echo /cmd              # surge + yaw-rate command
```

## Gazebo demo

The same head-on encounter with Gazebo (Harmonic) as the plant instead of
`fake_world` — the controller node is untouched; only the launch wiring
differs:

```bash
sudo apt install ros-jazzy-ros-gz      # once
ros2 launch colav_ros gazebo_demo.launch.py
```

<p align="center">
  <img src="../assets/gazebo_demo_head_on.gif" alt="Gazebo head-on demo" width="80%"/>
</p>

- [`worlds/colav_demo.sdf`](src/colav_ros/worlds/colav_demo.sdf): two kinematic
  vessels (`VelocityControl` + `OdometryPublisher` plugins), gravity off, no
  sensors or ocean shader — the server runs headless under software rendering.
- `ros_gz_bridge` carries `/clock`, both odometries (Gazebo → ROS) and both
  velocity commands (ROS → Gazebo); the controller's ego I/O is pure topic
  remapping, and `gz_obstacles` converts the obstacle vessel's odometry into
  the `ObstacleArray` the controller consumes.
- With `use_sim_time` the control timer ticks on `/clock`, so Gazebo owns the
  clock. The run completes the identical avoid/hold/resume cycle, reaching the
  goal at t = 8.45 s sim vs 8.40 s under fake_world (10 ms physics vs the
  fake world's 50 ms Euler steps).

## Status

- [x] ROS 2 interface + controller node (Python `rclpy`), runnable end-to-end
- [x] C++ `rclcpp` node (`colav_node_cpp`) linking `colav_core`, a verified
      C++ reimplementation cross-checked bit-identical to the Python controller
      (full-trajectory cross-check passes); runs end-to-end against `fake_world`
- [x] Gazebo world in place of `fake_world` (headless Harmonic server via
      `ros_gz_bridge`, sim-time control loop, controller unchanged)
- [ ] Hardware-in-the-loop: the C++ node on an embedded target driving the
      same topic contract
