"""
Adapter between a Gazebo world and the COLAV controller's obstacle input.

Two jobs, both on the world side of the topic contract (the controller
node is unaware Gazebo is involved — its ego I/O is wired straight
through ros_gz_bridge by topic remapping in gazebo_demo.launch.py):

- converts the obstacle vessel's bridged odometry into the
  colav_interfaces/ObstacleArray the controller consumes: position from
  the pose, heading from the quaternion, speed from the body-frame
  forward velocity (OdometryPublisher reports twist in the child frame);
- keeps the obstacle vessel sailing by re-publishing its constant
  surge command at 1 Hz (gz VelocityControl holds the last command, the
  periodic re-send makes the demo robust to a dropped first message).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from colav_interfaces.msg import Obstacle, ObstacleArray

from colav_ros.util import yaw_from_quaternion


class GzObstacles(Node):
    def __init__(self):
        super().__init__('gz_obstacles')
        self.declare_parameter('obstacle_speed', 2.0)
        self._speed = self.get_parameter('obstacle_speed').value

        self.create_subscription(
            Odometry, '/model/obstacle/odometry', self._on_odom,
            qos_profile_sensor_data)
        self._obstacles_pub = self.create_publisher(ObstacleArray, 'obstacles', 10)
        self._drive_pub = self.create_publisher(Twist, '/model/obstacle/cmd_vel', 10)

        self.create_timer(1.0, self._drive)

    def _on_odom(self, msg: Odometry):
        o = Obstacle()
        o.x = msg.pose.pose.position.x
        o.y = msg.pose.pose.position.y
        o.heading = yaw_from_quaternion(msg.pose.pose.orientation)
        o.velocity = msg.twist.twist.linear.x  # body-frame forward speed

        out = ObstacleArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'map'
        out.obstacles.append(o)
        self._obstacles_pub.publish(out)

    def _drive(self):
        cmd = Twist()
        cmd.linear.x = self._speed
        self._drive_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GzObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
