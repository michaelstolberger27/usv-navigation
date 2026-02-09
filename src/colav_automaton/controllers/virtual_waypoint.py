"""
Virtual waypoint computation for COLREGs-compliant collision avoidance.

Selects V1 (starboard-most unsafe set vertex ahead of ship) and applies
optional buffering for extra safety margin.
"""

from typing import List, Tuple, Optional, Callable
import numpy as np


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _point_in_polygon(px: float, py: float, vertices: List[Tuple[float, float]]) -> bool:
    """Check if point (px, py) is inside a polygon using ray casting."""
    if len(vertices) < 3:
        return False

    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid (geometric center) of a polygon."""
    if not vertices:
        return (0.0, 0.0)

    centroid_x = sum(v[0] for v in vertices) / len(vertices)
    centroid_y = sum(v[1] for v in vertices) / len(vertices)
    return (centroid_x, centroid_y)


def _offset_point_from_centroid(
    point_x: float,
    point_y: float,
    centroid_x: float,
    centroid_y: float,
    offset_distance: float
) -> Tuple[float, float]:
    """Move a point outward from a centroid by a specified distance."""
    outward = np.array([point_x - centroid_x, point_y - centroid_y])
    outward_len = np.linalg.norm(outward)

    if outward_len < 1e-6:
        return (point_x, point_y)

    outward_normalized = outward / outward_len
    new_x = point_x + offset_distance * outward_normalized[0]
    new_y = point_y + offset_distance * outward_normalized[1]

    return (new_x, new_y)


def default_vertex_provider(
    pos_x: float,
    pos_y: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float,
    psi: float = 0.0
) -> Optional[List[Tuple[float, float]]]:
    """
    Default vertex provider using circular obstacle approximation.

    Creates 8 vertices around each obstacle at distance Cs.

    Args:
        pos_x, pos_y: Ship position (unused in circular approximation)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius around obstacles
        psi: Ship heading (unused in circular approximation)

    Returns:
        List of (vx, vy) vertices or None if no obstacles
    """
    if not obstacles_list:
        return None

    vertices = []
    for ox, oy, _, _ in obstacles_list:
        for i in range(8):
            angle = i * np.pi / 4
            vertices.append((ox + Cs * np.cos(angle), oy + Cs * np.sin(angle)))

    return vertices if vertices else None


def compute_v1(
    pos_x: float,
    pos_y: float,
    psi: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float,
    vertex_provider: Callable[
        [float, float, List[Tuple[float, float, float, float]], float, float],
        Optional[List[Tuple[float, float]]]
    ],
    buffer_distance: float = 0.0
) -> Optional[Tuple[float, float]]:
    """
    Compute virtual waypoint V1 (starboard-most vertex ahead of ship).

    Selects the unsafe set vertex that is:
    1. Within +/-90 deg of ship heading ("ahead")
    2. Starboard-most (most negative relative angle among ahead vertices)
    3. Optionally applies outward buffer for extra safety margin

    Args:
        pos_x, pos_y: Current ship position
        psi: Current heading (radians)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety distance from obstacles
        vertex_provider: Function to generate unsafe set vertices
        buffer_distance: Optional buffer distance to apply (default 0.0)

    Returns:
        Tuple (v1_x, v1_y) or None if no valid vertex ahead
    """
    if not obstacles_list:
        return None

    vertices = vertex_provider(pos_x, pos_y, obstacles_list, Cs, psi)
    if not vertices:
        return None

    best_vertex = None
    best_angle = np.inf

    for vx, vy in vertices:
        angle_to_vertex = np.arctan2(vy - pos_y, vx - pos_x)
        relative_angle = _normalize_angle(angle_to_vertex - psi)

        # Only vertices ahead (within +/-pi/2 of heading)
        if -np.pi/2 < relative_angle < np.pi/2:
            # Prefer starboard (most negative angle)
            if relative_angle < best_angle:
                best_angle = relative_angle
                best_vertex = (vx, vy)

    if best_vertex is None:
        return None

    if buffer_distance > 0:
        best_vertex = _apply_v1_buffer(
            best_vertex[0], best_vertex[1], vertices, obstacles_list,
            buffer_distance
        )

    return best_vertex


def _apply_v1_buffer(
    v1_x: float,
    v1_y: float,
    vertices: List[Tuple[float, float]],
    obstacles_list: List[Tuple[float, float, float, float]],
    buffer_distance: float,
) -> Tuple[float, float]:
    """
    Apply buffer to V1 by moving it outward from polygon centroid.

    The buffer is rejected (original V1 returned) if:
    - Buffered V1 would be inside the unsafe polygon
    - Buffered V1 would be closer to any obstacle than original V1
    """
    if buffer_distance <= 0 or not vertices or len(vertices) < 3:
        return (v1_x, v1_y)

    centroid_x, centroid_y = _polygon_centroid(vertices)
    buffered_x, buffered_y = _offset_point_from_centroid(
        v1_x, v1_y, centroid_x, centroid_y, buffer_distance
    )

    # Reject if offset failed
    if (buffered_x, buffered_y) == (v1_x, v1_y):
        return (v1_x, v1_y)

    # Reject if buffered V1 would be inside polygon
    if _point_in_polygon(buffered_x, buffered_y, vertices):
        return (v1_x, v1_y)

    # Reject if buffered V1 is closer to any obstacle
    for ox, oy, _, _ in obstacles_list:
        orig_dist = np.hypot(v1_x - ox, v1_y - oy)
        buffered_dist = np.hypot(buffered_x - ox, buffered_y - oy)
        if buffered_dist < orig_dist - 0.1:
            return (v1_x, v1_y)

    return (buffered_x, buffered_y)
