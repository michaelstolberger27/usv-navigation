"""
Unsafe set computation and line-of-sight geometry utilities.

Handles convex hull generation, LOS cone creation, and multi-obstacle
collision region computation using the colav_unsafe_set API.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, Point
from colav_unsafe_set import create_unsafe_set, calculate_obstacle_metrics_for_agent
from colav_unsafe_set.objects import Agent, DynamicObstacle
from colav_unsafe_set.position_prediction import predict_position

logger = logging.getLogger(__name__)


def _create_agent(
    pos_x: float, pos_y: float, psi: float, v: float, Cs: float
) -> Agent:
    """Create an Agent object with heading converted to quaternion."""
    return Agent(
        position=(pos_x, pos_y, 0.0),
        orientation=(0.0, 0.0, np.sin(psi / 2), np.cos(psi / 2)),
        velocity=v,
        yaw_rate=0.0,
        safety_radius=Cs
    )


def _create_obstacles(
    obstacles_list: List[Tuple[float, float, float, float]], Cs: float
) -> List[DynamicObstacle]:
    """Create DynamicObstacle objects with headings converted to quaternions."""
    return [
        DynamicObstacle(
            tag=f"obs_{i}",
            position=(ox, oy, 0.0),
            orientation=(0.0, 0.0, np.sin(o_psi / 2), np.cos(o_psi / 2)),
            velocity=ov,
            yaw_rate=0.0,
            safety_radius=Cs
        )
        for i, (ox, oy, ov, o_psi) in enumerate(obstacles_list)
    ]


def _compute_swept_obstacles(
    agent: Agent,
    dynamic_obstacles: List[DynamicObstacle],
    ship_x: float,
    ship_v: float,
) -> List[DynamicObstacle]:
    """
    Expand obstacle list with predicted future positions to create a swept region.

    For moving obstacles, adds predicted positions at 50% and 100% of estimated
    maneuver time, ensuring the unsafe set covers the obstacle's trajectory.
    """
    obstacles = []
    try:
        metrics = calculate_obstacle_metrics_for_agent(agent, dynamic_obstacles)

        for obs_metric in metrics:
            obs = obs_metric.dynamic_obstacle
            tcpa = obs_metric.tcpa

            obstacles.append(obs)

            if obs.velocity > 0.1 and tcpa > 0:
                dist_to_obs = np.hypot(
                    obs.position[0] - agent.position[0],
                    obs.position[1] - agent.position[1]
                )
                maneuver_time = min(
                    max(tcpa, dist_to_obs / ship_v) * 1.5,
                    30.0
                )

                for dt in [maneuver_time * 0.5, maneuver_time]:
                    predicted_pos = predict_position(
                        obs.position, obs.orientation,
                        obs.velocity, obs.yaw_rate, dt
                    )
                    obstacles.append(DynamicObstacle(
                        tag=f"{obs.tag}_t{dt:.1f}",
                        position=(predicted_pos[0], predicted_pos[1], 0.0),
                        orientation=obs.orientation,
                        velocity=0.0,
                        yaw_rate=0.0,
                        safety_radius=obs.safety_radius
                    ))
    except Exception:
        logger.warning("Swept region prediction failed, using current positions")
        return dynamic_obstacles

    return obstacles


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
        logger.warning("Failed to create polygon from %d vertices", len(vertices))
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

    distance_safety = dsf if dsf is not None else Cs
    agent = _create_agent(ship_x, ship_y, ship_psi, ship_v, Cs)
    dynamic_obstacles = _create_obstacles(obstacles_list, Cs)

    if use_swept_region:
        obstacles_for_computation = _compute_swept_obstacles(
            agent, dynamic_obstacles, ship_x, ship_v
        )
    else:
        obstacles_for_computation = dynamic_obstacles

    try:
        convex_hull_vertices = create_unsafe_set(
            agent, obstacles_for_computation, float(distance_safety)
        )
        return convex_hull_vertices if convex_hull_vertices else None
    except Exception:
        logger.warning("create_unsafe_set failed for %d obstacles", len(obstacles_list))
        return None


def create_los_cone(pos_x: float, pos_y: float, xw: float, yw: float, v: float, tp: float) -> Polygon:
    """
    Create LOS cone F(p(t)) = conv(B_2(p(t), vtp), pw) as a Shapely Polygon.

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
        return Point(pos_x, pos_y).buffer(0.01)

    unit_to_waypoint = ship_to_waypoint / dist_to_waypoint
    perp_vector = np.array([-unit_to_waypoint[1], unit_to_waypoint[0]])
    radius = v * tp

    left_point = np.array([pos_x, pos_y]) + radius * perp_vector
    right_point = np.array([pos_x, pos_y]) - radius * perp_vector

    return Polygon([tuple(left_point), tuple(right_point), (xw, yw)])


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

    distance_safety = Cs
    agent = _create_agent(pos_x, pos_y, ship_psi, ship_v, Cs)
    dynamic_obstacles = _create_obstacles(obstacles_list, Cs)
    obstacles_for_computation = _compute_swept_obstacles(
        agent, dynamic_obstacles, pos_x, ship_v
    )

    try:
        convex_hull_vertices = create_unsafe_set(
            agent, obstacles_for_computation, float(distance_safety)
        )
        return generate_dynamic_unsafe_set_from_vertices(convex_hull_vertices) if convex_hull_vertices else None
    except Exception:
        logger.warning("create_unsafe_set failed for %d obstacles", len(obstacles_list))
        return None


def check_collision_threat(
    pos_x: float,
    pos_y: float,
    psi: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    ship_v: float,
    Cs: float,
    dsafe: float
) -> bool:
    """
    Check if any obstacle poses a collision threat using DCPA/TCPA.

    An obstacle threatens if:
    - Time to Closest Point of Approach (TCPA) <= dsafe / ship_v
    - Distance at CPA (DCPA) <= dsafe

    Args:
        pos_x, pos_y: Current ship position
        psi: Current ship heading (rad)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        ship_v: Ship velocity (m/s)
        Cs: Safety radius (m)
        dsafe: Safe maneuvering distance (m)

    Returns:
        bool: True if any obstacle is a collision threat
    """
    if not obstacles_list:
        return False

    agent = _create_agent(pos_x, pos_y, psi, ship_v, Cs)
    dynamic_obstacles = _create_obstacles(obstacles_list, Cs)

    results = calculate_obstacle_metrics_for_agent(agent, dynamic_obstacles)
    tcpa_threshold = dsafe / ship_v

    return any(r.tcpa <= tcpa_threshold and r.dcpa <= dsafe for r in results)
