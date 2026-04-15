"""
Virtual waypoint computation for COLREGs-compliant collision avoidance.

Implements the V1 selection from the paper (Section 4.1, 4.3):
- V1 is chosen as the vertex of the unsafe set with the largest relative
  heading angle toward starboard (or port for rule 17b).
- Both starboard and port candidates are evaluated; the one yielding the
  larger predicted CPA is selected.
"""

from typing import List, Tuple, Optional, Callable
import numpy as np


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _predicted_cpa(pos_x, pos_y, v, heading, ox, oy, ovx, ovy):
    """Predict CPA distance if ego heads in the given direction."""
    new_vx = v * np.cos(heading)
    new_vy = v * np.sin(heading)
    dx, dy = ox - pos_x, oy - pos_y
    dvx = ovx - new_vx
    dvy = ovy - new_vy
    dv_sq = dvx ** 2 + dvy ** 2
    if dv_sq < 1e-6:
        return np.hypot(dx, dy)
    t = max(-(dx * dvx + dy * dvy) / dv_sq, 0.0)
    cx = pos_x + new_vx * t
    cy = pos_y + new_vy * t
    return np.hypot(cx - (ox + ovx * t), cy - (oy + ovy * t))


def _min_cpa_for_vertex(pos_x, pos_y, v, vx, vy, obstacles_list):
    """Return minimum predicted CPA across all obstacles if heading toward (vx, vy)."""
    heading = np.arctan2(vy - pos_y, vx - pos_x)
    min_cpa = np.inf
    for ox, oy, ov, o_psi in obstacles_list:
        cpa = _predicted_cpa(pos_x, pos_y, v, heading,
                             ox, oy, ov * np.cos(o_psi), ov * np.sin(o_psi))
        min_cpa = min(min_cpa, cpa)
    return min_cpa


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
    buffer_distance: float = 0.0,
    v: float = 12.0,
) -> Optional[Tuple[float, float]]:
    """
    Compute virtual waypoint V1 from unsafe set vertices (paper Section 4.1).

    Per the paper, V1 is the vertex of the unsafe set for which the relative
    heading angle ρ(V_i, p) is largest.  Starboard vertices (V1, V2 in the
    paper) have negative relative angles; port vertices (V3, V4) have positive.

    This implementation evaluates BOTH the starboard-most and port-most ahead
    vertices, predicts the CPA each would yield, and selects the one with the
    larger CPA.  This extends the paper to handle rule 17b (stand-on vessel
    taking late action to port when give-way vessel fails to act).

    Args:
        pos_x, pos_y: Current ship position
        psi: Current heading (radians)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety distance from obstacles
        vertex_provider: Function to generate unsafe set vertices
        buffer_distance: Optional buffer distance to apply (default 0.0)
        v: Ship velocity for CPA prediction (m/s)

    Returns:
        Tuple (v1_x, v1_y) or None if no valid vertex ahead
    """
    if not obstacles_list:
        return None

    vertices = vertex_provider(pos_x, pos_y, obstacles_list, Cs, psi)
    if not vertices:
        return None

    # Find starboard-most and port-most ahead vertices (paper Section 4.1)
    stbd_vertex = None
    stbd_angle = np.inf  # most negative = most starboard
    port_vertex = None
    port_angle = -np.inf  # most positive = most port

    for vx, vy in vertices:
        angle_to_vertex = np.arctan2(vy - pos_y, vx - pos_x)
        relative_angle = _normalize_angle(angle_to_vertex - psi)

        # Only vertices ahead (within ±90° of heading, paper eq 16 criterion)
        if -np.pi / 2 < relative_angle < np.pi / 2:
            # Starboard: most negative relative angle
            if relative_angle < stbd_angle:
                stbd_angle = relative_angle
                stbd_vertex = (vx, vy)
            # Port: most positive relative angle
            if relative_angle > port_angle:
                port_angle = relative_angle
                port_vertex = (vx, vy)

    # Evaluate CPA for both candidates
    stbd_cpa = (_min_cpa_for_vertex(pos_x, pos_y, v,
                                     stbd_vertex[0], stbd_vertex[1], obstacles_list)
                if stbd_vertex is not None else -1.0)
    port_cpa = (_min_cpa_for_vertex(pos_x, pos_y, v,
                                     port_vertex[0], port_vertex[1], obstacles_list)
                if port_vertex is not None else -1.0)

    if stbd_vertex is None and port_vertex is None:
        return None

    # COLREGs preference: choose starboard unless port gives significantly
    # better CPA (> 10% improvement).  This ensures starboard is the default
    # per Rules 13-15, while allowing port (rule 17b) when it's clearly better.
    if stbd_vertex is not None and (port_vertex is None or port_cpa <= stbd_cpa * 1.1):
        best_vertex = stbd_vertex
    else:
        best_vertex = port_vertex

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
