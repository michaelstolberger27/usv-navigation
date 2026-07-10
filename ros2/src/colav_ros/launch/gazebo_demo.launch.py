"""
Launch the COLAV controller against a Gazebo world.

    ros2 launch colav_ros gazebo_demo.launch.py

Same head-on encounter as demo.launch.py, but the plant is Gazebo
(server-only, headless) instead of fake_world:

- gz sim runs worlds/colav_demo.sdf: two kinematic vessels with
  VelocityControl + OdometryPublisher plugins, gravity off;
- ros_gz_bridge carries /clock, both odometries (gz -> ROS) and both
  velocity commands (ROS -> gz);
- colav_node is untouched — its ego_odom/cmd topics are simply remapped
  onto the bridged Gazebo topics, and use_sim_time makes its control
  timer tick on /clock;
- gz_obstacles adapts the obstacle vessel's odometry into the
  ObstacleArray the controller consumes and keeps that vessel sailing.

Swapping fake_world for Gazebo is exactly this launch file — no
controller change — which is the point of the topic contract.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    world = os.path.join(
        get_package_share_directory('colav_ros'), 'worlds', 'colav_demo.sdf')

    return LaunchDescription([
        # Gazebo server, headless, running immediately
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '-s', world],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/model/ego_vessel/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/obstacle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/ego_vessel/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/model/obstacle/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            ],
            output='screen',
        ),
        Node(
            package='colav_ros',
            executable='colav_node',
            name='colav_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[
                ('ego_odom', '/model/ego_vessel/odometry'),
                ('cmd', '/model/ego_vessel/cmd_vel'),
            ],
        ),
        Node(
            package='colav_ros',
            executable='gz_obstacles',
            name='gz_obstacles',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
