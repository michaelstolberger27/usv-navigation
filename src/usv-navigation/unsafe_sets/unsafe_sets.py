"""
Unsafe set computation and line-of-sight geometry utilities.

Handles convex hull generation, LOS cone creation, and multi-obstacle
collision region computation using the colav_unsafe_set API.
"""

from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from colav_unsafe_set import create_unsafe_set
from colav_unsafe_set.objects import Agent, DynamicObstacle


def generate_dynamic_unsafe_set_from_vertices(vertices: List[List[float]]) -> Optional[Polygon]:
    """
    Create a Shapely Polygon from dynamic unsafe set vertices (convex hull).

    Args:
        vertices: List of [x, y] coordinate pairs representing convex hull vertices

    Returns:
        Polygon: Unsafe set polygon, or None if vertices insufficient
    """
    if not vertices or len(vertices) < 3:
        return None
    
    try:
        return Polygon(vertices)
    except Exception:
        return None


def get_unsafe_set_vertices(
    ship_x: float, 
    ship_y: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float,
    dsf: Optional[float] = None,
    time_of_interest: float = 10.0
) -> Optional[List[List[float]]]:
    """
    Generate dynamic unsafe set vertices using the unsafe-set API.
    
    Uses the colav_unsafe_set package to compute unsafe regions based on
    dynamic obstacle metrics (DCPA, TCPA) rather than simple circles.

    Args:
        ship_x, ship_y: Ship position
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius (used as default if dsf not provided)
        dsf: Distance safety threshold (defaults to Cs if not provided)
        time_of_interest: Unused parameter kept for compatibility

    Returns:
        List[List[float]]: Convex hull vertices of unsafe regions, or None if empty
    """
    if not obstacles_list:
        return None
    
    # Use provided dsf or default to Cs
    distance_safety = dsf if dsf is not None else Cs
    
    # Create Agent object (ship)
    agent = Agent(
        position=(float(ship_x), float(ship_y), 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
        velocity=12.0,  # Typical ship velocity
        yaw_rate=0.0,
        safety_radius=float(Cs)
    )
    
    # Create DynamicObstacle objects
    dynamic_obstacles = []
    for i, (ox, oy, ov, o_psi) in enumerate(obstacles_list):
        # Convert yaw angle to quaternion (rotation around z-axis)
        qx, qy = 0.0, 0.0
        qz = np.sin(float(o_psi) / 2.0)
        qw = np.cos(float(o_psi) / 2.0)
        
        obstacle = DynamicObstacle(
            tag=f"obstacle_{i}",
            position=(float(ox), float(oy), 0.0),
            orientation=(qx, qy, qz, qw),
            velocity=float(ov),
            yaw_rate=0.0,
            safety_radius=float(Cs)
        )
        dynamic_obstacles.append(obstacle)
    
    try:
        # Use the unsafe-set API to compute and return convex hull vertices
        # create_unsafe_set internally calls: metrics → I1,I2,I3 → unionize → gen_uIoI_convhull
        convex_hull_vertices = create_unsafe_set(agent, dynamic_obstacles, float(distance_safety))
        
        # Return vertices if we found any
        return convex_hull_vertices if convex_hull_vertices else None
        
    except Exception as e:
        # API call failed - return None
        return None


def create_los_cone(pos_x: float, pos_y: float, xw: float, yw: float, v: float, tp: float) -> Polygon:
    """
    Create LOS cone F(p(t)) = conv(B₂(p(t), vtp), pw) as a Shapely Polygon.

    The cone is formed by the convex hull of:
    - A circle of radius vtp around current position
    - The waypoint

    We approximate the circle with vertices perpendicular to the LOS direction.

    Args:
        pos_x, pos_y: Current ship position
        xw, yw: Waypoint position
        v: Ship velocity
        tp: Prescribed time

    Returns:
        Polygon: LOS cone as convex polygon
    """
    ship_to_waypoint = np.array([xw - pos_x, yw - pos_y])
    dist_to_waypoint = np.linalg.norm(ship_to_waypoint)

    if dist_to_waypoint < 1e-6:
        # At waypoint, return point
        return Point(pos_x, pos_y).buffer(0.01)

    # Unit vector toward waypoint
    unit_to_waypoint = ship_to_waypoint / dist_to_waypoint

    # Perpendicular vector (rotate 90 degrees)
    perp_vector = np.array([-unit_to_waypoint[1], unit_to_waypoint[0]])

    # Radius of uncertainty circle
    radius = v * tp

    # Create cone: left edge, ship position with radius, right edge, waypoint
    left_point = np.array([pos_x, pos_y]) + radius * perp_vector
    right_point = np.array([pos_x, pos_y]) - radius * perp_vector

    # Convex hull forms a triangle/cone
    cone_points = [
        tuple(left_point),
        tuple(right_point),
        (xw, yw)
    ]

    return Polygon(cone_points)


def compute_unified_unsafe_region(
    pos_x: float,
    pos_y: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float
) -> Optional[Polygon]:
    """
    Compute unified unsafe region from all obstacles (multi-obstacle optimization).
    
    For crowded scenarios (e.g., Scenario 3), computes a single unified convex hull
    covering all obstacles instead of checking each separately. This enables optimal
    path planning in complex environments.

    Args:
        pos_x, pos_y: Ship position
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius

    Returns:
        Polygon: Unified unsafe set, or None if no obstacles
    """
    if not obstacles_list or len(obstacles_list) < 1:
        return None
    
    # Collect vertices from all obstacles
    all_vertices = []
    
    for ox, oy, ov, o_psi in obstacles_list:
        # Get unsafe set for this obstacle
        vertices = get_unsafe_set_vertices(pos_x, pos_y, [(ox, oy, ov, o_psi)], Cs)
        if vertices:
            all_vertices.extend(vertices)
    
    if len(all_vertices) < 3:
        return None
    
    # Create unified convex hull
    try:
        return generate_dynamic_unsafe_set_from_vertices(all_vertices)
    except Exception:
        return None
