"""
Launch-testing smoke test: colav_node + fake_world end to end.

Brings up the same node pair as demo.launch.py and asserts, from the
outside via the colav_state topic, that the controller completes the
avoidance cycle: COLLISION_AVOIDANCE -> CONSTANT_CONTROL ->
WAYPOINT_REACHING. This is the ROS-level counterpart of the core
behavioural regression suite — it exercises discovery, QoS matching,
message conversion, and the timer loop rather than the algorithm.

No exit-code assertions on purpose: rclpy teardown under launch_testing's
SIGINT is racy by design and says nothing about the system under test.
"""

import os
import time
import unittest
from pathlib import Path

import launch
import launch_ros.actions
import launch_testing.actions
import pytest

import rclpy
from std_msgs.msg import String

# The controller node imports the simulator-independent core from the
# repo's src/ via COLAV_REPO (see colav_node.py). The repo root is three
# levels up from this package: ros2/src/colav_ros/test/ -> repo.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])

CYCLE = ["COLLISION_AVOIDANCE", "CONSTANT_CONTROL", "WAYPOINT_REACHING"]


@pytest.mark.launch_test
def generate_test_description():
    return launch.LaunchDescription([
        launch.actions.SetEnvironmentVariable('COLAV_REPO', _REPO_ROOT),
        launch_ros.actions.Node(
            package='colav_ros', executable='colav_node',
            name='colav_node', output='screen'),
        launch_ros.actions.Node(
            package='colav_ros', executable='fake_world',
            name='fake_world', output='screen'),
        launch_testing.actions.ReadyToTest(),
    ])


def _compress(labels):
    """Collapse consecutive duplicates: S1 S1 S2 S2 S3 -> S1 S2 S3."""
    out = []
    for label in labels:
        if not out or out[-1] != label:
            out.append(label)
    return out


def _contains_cycle(compressed):
    """True if `compressed` contains CYCLE as a contiguous subsequence."""
    n = len(CYCLE)
    return any(compressed[i:i + n] == CYCLE
               for i in range(len(compressed) - n + 1))


class TestAvoidanceCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('smoke_test_listener')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_full_avoidance_cycle_observed(self):
        # The demo scenario completes the cycle in ~6 s of sim time
        # (which is wall time here); the generous deadline absorbs
        # discovery latency and slow CI runners.
        states = []
        self.node.create_subscription(
            String, 'colav_state', lambda m: states.append(m.data), 10)

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if _contains_cycle(_compress(states)):
                break

        compressed = _compress(states)
        self.assertTrue(
            _contains_cycle(compressed),
            f"avoid->hold->resume cycle not observed within 60 s; "
            f"state sequence seen: {compressed} "
            f"(COLAV_REPO={os.environ.get('COLAV_REPO')})")
