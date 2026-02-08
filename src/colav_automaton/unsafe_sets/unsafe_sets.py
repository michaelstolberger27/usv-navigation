"""
Unsafe set computation and line-of-sight geometry utilities.

Handles convex hull generation, LOS cone creation, and multi-obstacle
collision region computation using the colav_unsafe_set API.
"""

from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from colav_unsafe_set import create_unsafe_set, calculate_obstacle_metrics_for_agent
from colav_unsafe_set.objects import Agent, DynamicObstacle
from colav_unsafe_set.position_prediction import predict_position


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
    ship_psi: float = 0.0,
    ship_v: float = 12.0,
    use_swept_region: bool = True
) -> Optional[List[List[float]]]:
    """
    Generate dynamic unsafe set vertices using the unsafe-set API.

    Uses the colav_unsafe_set package to compute unsafe regions based on
    dynamic obstacle metrics (DCPA, TCPA) rather than simple circles.

    For moving obstacles, can predict future positions to create a "swept"
    region covering the obstacle's trajectory during the maneuver.

    Args:
        ship_x, ship_y: Ship position
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius (used as default if dsf not provided)
        dsf: Distance safety threshold (defaults to Cs if not provided)
        ship_psi: Ship heading in radians (default 0.0)
        ship_v: Ship velocity in m/s (default 12.0)
        use_swept_region: If True, creates swept region for moving obstacles
            (use for V1 computation). If False, uses current positions only
            (use for G11 guard checks).

    Returns:
        List[List[float]]: Convex hull vertices of unsafe regions, or None if empty
    """
    if not obstacles_list:
        return None

    # Use provided dsf or default to Cs
    distance_safety = dsf if dsf is not None else Cs

    # Convert ship heading to quaternion (rotation around z-axis)
    ship_qz = np.sin(float(ship_psi) / 2.0)
    ship_qw = np.cos(float(ship_psi) / 2.0)

    # Create Agent object (ship) with correct heading
    agent = Agent(
        position=(float(ship_x), float(ship_y), 0.0),
        orientation=(0.0, 0.0, ship_qz, ship_qw),
        velocity=float(ship_v),
        yaw_rate=0.0,
        safety_radius=float(Cs)
    )

    # Create DynamicObstacle objects with current positions
    dynamic_obstacles = []
    for i, (ox, oy, ov, o_psi) in enumerate(obstacles_list):
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

    # Determine which obstacles to use for unsafe set computation
    if use_swept_region:
        # For V1 computation: create "swept" region covering current and future positions
        # This ensures V1 accounts for where the obstacle will be during the maneuver
        obstacles_for_computation = []
        try:
            obstacles_with_metrics = calculate_obstacle_metrics_for_agent(agent, dynamic_obstacles)

            for obs_metric in obstacles_with_metrics:
                obs = obs_metric.dynamic_obstacle
                tcpa = obs_metric.tcpa

                # Always include current position
                obstacles_for_computation.append(obs)

                # For moving obstacles, add predicted positions along trajectory
                if obs.velocity > 0.1 and tcpa > 0:
                    # Estimate maneuver time based on distance and ship speed
                    dist_to_obs = np.sqrt(
                        (obs.position[0] - ship_x)**2 +
                        (obs.position[1] - ship_y)**2
                    )
                    maneuver_time = max(tcpa, dist_to_obs / ship_v) * 1.5  # Add 50% margin
                    maneuver_time = min(maneuver_time, 30.0)  # Cap at 30 seconds

                    # Add predicted positions at intervals to create swept region
                    for dt in [maneuver_time * 0.5, maneuver_time]:
                        predicted_pos = predict_position(
                            obs.position,
                            obs.orientation,
                            obs.velocity,
                            obs.yaw_rate,
                            dt
                        )
                        predicted_obstacle = DynamicObstacle(
                            tag=f"{obs.tag}_t{dt:.1f}",
                            position=(predicted_pos[0], predicted_pos[1], 0.0),
                            orientation=obs.orientation,
                            velocity=0.0,
                            yaw_rate=0.0,
                            safety_radius=obs.safety_radius
                        )
                        obstacles_for_computation.append(predicted_obstacle)
        except Exception:
            # Fall back to current positions if prediction fails
            obstacles_for_computation = dynamic_obstacles
    else:
        # For G11 checks: use current obstacle positions only
        obstacles_for_computation = dynamic_obstacles

    try:
        # Compute unsafe set
        convex_hull_vertices = create_unsafe_set(agent, obstacles_for_computation, float(distance_safety))

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
    Cs: float,
    ship_psi: float = 0.0,
    ship_v: float = 12.0
) -> Optional[Polygon]:
    """
    Compute unified unsafe region from all obstacles for G11 guard checks.

    Uses swept regions to account for obstacle motion, ensuring the unsafe
    set covers where dynamic obstacles will be during the maneuver window.

    Args:
        pos_x, pos_y: Ship position
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius
        ship_psi: Ship heading (rad)
        ship_v: Ship velocity (m/s)

    Returns:
        Polygon: Unified unsafe set, or None if no obstacles
    """
    if not obstacles_list:
        return None

    all_vertices = []

    for ox, oy, ov, o_psi in obstacles_list:
        vertices = get_unsafe_set_vertices(
            pos_x, pos_y, [(ox, oy, ov, o_psi)], Cs,
            ship_psi=ship_psi, ship_v=ship_v,
            use_swept_region=True
        )
        if vertices:
            all_vertices.extend(vertices)

    if len(all_vertices) < 3:
        return None

    try:
        return generate_dynamic_unsafe_set_from_vertices(all_vertices)
    except Exception:
        return None
