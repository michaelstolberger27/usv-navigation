"""
Launch the COLAV node together with the closed-loop fake world.

    ros2 launch colav_ros demo.launch.py

Brings up colav_node (the controller) and fake_world (a stand-in plant +
traffic). They are wired by the default topic names — ego_odom, obstacles,
cmd — so swapping fake_world for a VRX/Gazebo bridge later needs no change
to the controller node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='colav_ros',
            executable='colav_node',
            name='colav_node',
            output='screen',
        ),
        Node(
            package='colav_ros',
            executable='fake_world',
            name='fake_world',
            output='screen',
        ),
    ])
