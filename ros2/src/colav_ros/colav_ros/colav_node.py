"""
ROS 2 node wrapping the COLAV hybrid automaton.

Architecture: asynchronous I/O at the edges (subscriptions update buffers
as messages arrive), deterministic control in the middle (a fixed-rate
timer reads the latest state and steps the tick-synchronous runtime once
per tick). This is the standard time-triggered control pattern and is
exactly what `SyncColavRuntime.step_external` was built for — the same
core that the CommonOcean adapter and the AIS replay runner drive, here
behind ROS topics instead.

Topics:
    sub  ego_odom   (nav_msgs/Odometry)              ego pose + heading
    sub  obstacles  (colav_interfaces/ObstacleArray) traffic (x, y, v, psi)
    pub  cmd        (geometry_msgs/Twist)            linear.x = surge speed,
                                                     angular.z = yaw rate
    pub  colav_state (std_msgs/String)               S1 / S2 / S3 label

The node does NOT integrate the vessel — whoever owns the plant (the
fake_world node here, VRX/Gazebo or real hardware later) integrates the
published command and feeds pose back on ego_odom.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from colav_interfaces.msg import ObstacleArray

# Import the simulator-independent core. ROS runs the system Python; if
# COLAV_REPO is set, put this repo's src/ first so it wins over any other
# colav_automaton that happens to be installed in the environment.
import os as _os
import sys as _sys
_repo = _os.environ.get("COLAV_REPO")
if _repo:
    _sys.path.insert(0, _os.path.join(_repo, "src"))
from colav_automaton import SyncColavRuntime  # noqa: E402


def _yaw_from_quaternion(q) -> float:
    """Extract the planar heading (yaw) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ColavNode(Node):
    def __init__(self):
        super().__init__('colav_node')

        # --- parameters (demo defaults match examples/realtime_simulation
        #     scenario 3: small-scale head-on, so K_off=1.0 pure G23 resume
        #     and tp_control wide enough for dt; see that file's rationale) ---
        self.declare_parameters('', [
            ('goal_x', 100.0), ('goal_y', 0.0),
            ('Cs', 5.0), ('v', 12.0), ('tp', 1.0),
            ('a', 1.67), ('eta', 3.5),
            ('dt', 0.05), ('K_off', 1.0), ('tp_control', 2.0),
            ('arrival_radius', 1.5),
        ])
        g = {name: self.get_parameter(name).value for name in (
            'goal_x', 'goal_y', 'Cs', 'v', 'tp', 'a', 'eta',
            'dt', 'K_off', 'tp_control', 'arrival_radius')}
        self._a = g['a']
        self._v = g['v']
        self._dt = g['dt']
        self._goal = (g['goal_x'], g['goal_y'])
        self._arrival_radius = g['arrival_radius']

        self._rt = SyncColavRuntime(
            waypoint=self._goal,
            obstacles=[],
            initial_state=(0.0, 0.0, 0.0),
            Cs=g['Cs'], v=g['v'], tp=g['tp'], a=g['a'], eta=g['eta'],
            K_off=g['K_off'], tp_control=g['tp_control'],
        )

        # Latest world state, written by the subscription callbacks and read
        # by the timer. Plain attributes are fine: rclpy's default executor
        # runs callbacks and the timer on one thread, so there is no race.
        self._ego = (0.0, 0.0, 0.0)
        self._obstacles = []
        self._arrived = False
        self._last_mode = None

        self.create_subscription(Odometry, 'ego_odom', self._on_odom, 10)
        self.create_subscription(ObstacleArray, 'obstacles', self._on_obstacles, 10)
        self._cmd_pub = self.create_publisher(Twist, 'cmd', 10)
        self._state_pub = self.create_publisher(String, 'colav_state', 10)

        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"colav_node up: goal={self._goal}, dt={self._dt}s, "
            f"Cs={g['Cs']}, v={g['v']} m/s")

    # ---- async edges: cache the latest world ----

    def _on_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        psi = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._ego = (pos.x, pos.y, psi)

    def _on_obstacles(self, msg: ObstacleArray):
        self._obstacles = [(o.x, o.y, o.velocity, o.heading) for o in msg.obstacles]

    # ---- deterministic core: one tick ----

    def _tick(self):
        if self._arrived:
            return

        x, y, psi = self._ego
        result = self._rt.step_external(self._dt, [x, y, psi],
                                        obstacles=self._obstacles)

        # Heading plant: yaw_rate = -a*psi + a*u (same as the CommonOcean
        # adapter); surge held at the cruise speed.
        yaw_rate = -self._a * psi + self._a * result.u
        cmd = Twist()
        cmd.linear.x = self._v
        cmd.angular.z = yaw_rate
        self._cmd_pub.publish(cmd)
        self._state_pub.publish(String(data=result.mode))

        if result.transition or result.mode != self._last_mode:
            self.get_logger().info(
                f"t={result.t:6.2f}s  {result.mode}"
                + (f"  [{result.transition}]" if result.transition else ""))
            self._last_mode = result.mode

        if math.hypot(self._goal[0] - x, self._goal[1] - y) < self._arrival_radius:
            self._arrived = True
            self._cmd_pub.publish(Twist())  # stop
            self.get_logger().info(f"goal reached at t={result.t:.2f}s")


def main(args=None):
    rclpy.init(args=args)
    node = ColavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
