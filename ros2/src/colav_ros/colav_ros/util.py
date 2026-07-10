"""Small helpers shared by the colav_ros nodes."""

import math


def yaw_from_quaternion(q) -> float:
    """Extract the planar heading (yaw) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
