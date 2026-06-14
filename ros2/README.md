# ROS 2 integration (`colav_ros`)

A ROS 2 node that wraps the COLAV hybrid automaton, plus a closed-loop
fake-world node so the full graph runs without a heavyweight simulator.
This is the **Phase 7** entry point (roadmap in `../HANDOFF.md` §4);
VRX/Gazebo and a C++ `rclcpp` node build on this foundation.

Tested on **ROS 2 Jazzy**.

## Packages

| Package | Build type | Contents |
|---|---|---|
| `colav_interfaces` | `ament_cmake` | `Obstacle.msg`, `ObstacleArray.msg` |
| `colav_ros` | `ament_python` | `colav_node` (controller), `fake_world` (stand-in plant + traffic), `demo.launch.py` |

## Design

Asynchronous I/O at the edges, deterministic control in the middle — the
standard time-triggered pattern:

- subscriptions cache the latest ego pose and obstacle list as messages arrive;
- a fixed-rate timer reads that snapshot and steps `SyncColavRuntime.step_external`
  once per tick (the same deterministic core the CommonOcean adapter and the
  AIS replay runner drive);
- the node publishes a `Twist` command and never integrates the vessel — the
  world on the other side (fake_world here, VRX/Gazebo or real hardware later)
  owns integration and feeds pose back.

```
            ego_odom (nav_msgs/Odometry)
 fake_world ───────────────────────────▶ colav_node
   (plant   ◀───────────────────────────   (SyncColavRuntime)
  + traffic)        cmd (geometry_msgs/Twist)
            obstacles (colav_interfaces/ObstacleArray) ▶
```

Because the controller depends only on the topic contract, swapping
`fake_world` for a VRX bridge is a launch-file change, not a code change.

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

## Status

- [x] ROS 2 interface + controller node (Python `rclpy`), runnable end-to-end
- [ ] VRX/Gazebo world in place of `fake_world`
- [ ] C++ `rclcpp` node (the targeted-C++ deliverable; decision pending —
      embed the Python core via pybind11 vs. a verified C++ reimplementation
      cross-checked against the Python reference)
