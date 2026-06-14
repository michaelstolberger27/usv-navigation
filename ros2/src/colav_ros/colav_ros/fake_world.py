"""
Minimal closed-loop world for demonstrating the COLAV node without a full
simulator. Integrates the ego from the controller's `cmd` and advances the
obstacles on straight-line constant-velocity tracks, publishing both back.

This stands in for VRX/Gazebo (next step): it speaks the same topics, so
the colav_node is unaware of which world is on the other side. Default
scenario is the small-scale head-on encounter (ego 0,0 -> goal 100,0;
obstacle approaching head-on from 70,0), matching example scenario 3.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from colav_interfaces.msg import Obstacle, ObstacleArray


def _quaternion_from_yaw(psi: float):
    """geometry_msgs Quaternion (as a 4-tuple) for a planar heading."""
    return (0.0, 0.0, math.sin(psi / 2.0), math.cos(psi / 2.0))


class FakeWorld(Node):
    def __init__(self):
        super().__init__('fake_world')

        self.declare_parameters('', [
            ('ego_x', 0.0), ('ego_y', 0.0), ('ego_psi', 0.0),
            ('dt', 0.05),
            # one obstacle: x, y, velocity, heading (head-on by default)
            ('obs_x', 70.0), ('obs_y', 0.0),
            ('obs_v', 2.0), ('obs_heading', math.pi),
        ])
        g = {n: self.get_parameter(n).value for n in (
            'ego_x', 'ego_y', 'ego_psi', 'dt',
            'obs_x', 'obs_y', 'obs_v', 'obs_heading')}

        self._dt = g['dt']
        self._ego = [g['ego_x'], g['ego_y'], g['ego_psi']]
        self._obs0 = (g['obs_x'], g['obs_y'], g['obs_v'], g['obs_heading'])
        self._t = 0.0
        self._cmd = (0.0, 0.0)  # (surge v, yaw_rate) — held until first cmd

        self.create_subscription(Twist, 'cmd', self._on_cmd, 10)
        self._odom_pub = self.create_publisher(Odometry, 'ego_odom', 10)
        self._obs_pub = self.create_publisher(ObstacleArray, 'obstacles', 10)
        self.create_timer(self._dt, self._step)

    def _on_cmd(self, msg: Twist):
        self._cmd = (msg.linear.x, msg.angular.z)

    def _step(self):
        v, yaw_rate = self._cmd
        x, y, psi = self._ego
        # Integrate the ego plant (Euler), then advance the obstacle.
        psi += yaw_rate * self._dt
        x += v * math.cos(psi) * self._dt
        y += v * math.sin(psi) * self._dt
        self._ego = [x, y, psi]

        ox, oy, ov, opsi = self._obs0
        ox = ox + ov * math.cos(opsi) * self._t
        oy = oy + ov * math.sin(opsi) * self._t
        self._t += self._dt

        self._publish_odom(x, y, psi)
        self._publish_obstacles([(ox, oy, ov, opsi)])

    def _publish_odom(self, x, y, psi):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'map'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        qx, qy, qz, qw = _quaternion_from_yaw(psi)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom)

    def _publish_obstacles(self, obstacles):
        msg = ObstacleArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for ox, oy, ov, opsi in obstacles:
            o = Obstacle()
            o.x, o.y, o.velocity, o.heading = ox, oy, ov, opsi
            msg.obstacles.append(o)
        self._obs_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeWorld()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
