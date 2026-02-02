"""
Collision detection and path planning condition checks.

Implements guard condition functions (G11, G12, L1, L2) that determine
when collision avoidance is needed and when avoidance maneuvers complete.
"""

from typing import List, Tuple
import numpy as np
from colav_unsafe_set import calculate_obstacle_metrics_for_agent
from colav_automaton.unsafe_sets import create_los_cone, compute_unified_unsafe_region


def check_G11_dynamic(
    pos_x: float,
    pos_y: float,
    xw: float,
    yw: float,
    v: float,
    tp: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float
) -> bool:
    """
    G11 (dynamic): Check if LOS to waypoint intersects any unsafe region.
    
    For multiple obstacles, uses unified convex hull (multi-obstacle optimization).
    Checks if line-of-sight cone intersects the combined unsafe region.

    Args:
        pos_x, pos_y: Current ship position
        xw, yw: Waypoint position
        v: Ship velocity
        tp: Prescribed time
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius for obstacles

    Returns:
        bool: True if LOS cone intersects any unsafe region
    """
    if not obstacles_list:
        return False

    # Use unified unsafe region for multi-obstacle scenarios
    unsafe_polygon = compute_unified_unsafe_region(pos_x, pos_y, obstacles_list, Cs)

    if unsafe_polygon is None:
        return False

    # Use a narrower LOS cone for checking (reduced uncertainty after maneuvers)
    # This prevents false positives when ship has already passed obstacle
    effective_tp = min(tp, 1.0)  # Cap cone width at 1 second of travel
    los_cone = create_los_cone(pos_x, pos_y, xw, yw, v, effective_tp)

    # Check intersection
    return unsafe_polygon.intersects(los_cone)


def check_G12_dynamic(
    pos_x: float,
    pos_y: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    dsafe: float,
    ship_v: float = 12.0,
    tp: float = 15.0,
    Cs: float = 2.0
) -> bool:
    """
    G12 (dynamic): Check if any obstacle poses threat using DCPA/TCPA metrics.
    
    Uses the unsafe-set API to compute collision prediction metrics instead of
    simple distance. An obstacle threatens if:
    - Time to Closest Point of Approach (TCPA) < prescribed time (tp)
    - Distance at CPA (DCPA) < safety radius (Cs)

    Args:
        pos_x, pos_y: Current ship position
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        dsafe: Safe distance threshold (for fallback)
        ship_v: Ship velocity (m/s)
        tp: Prescribed time lookahead (seconds)
        Cs: Safety radius (m)

    Returns:
        bool: True if any obstacle is a collision threat
    """
    if not obstacles_list:
        return False
    
    # Use API metrics for each obstacle
    for ox, oy, ov, o_psi in obstacles_list:
        try:
            metrics = calculate_obstacle_metrics_for_agent(
                agent_x=pos_x, agent_y=pos_y,
                obstacle_x=ox, obstacle_y=oy,
                obstacle_velocity=ov, obstacle_heading=o_psi,
                agent_velocity=ship_v, safety_radius=Cs
            )
            
            # Check if obstacle will be closest within our lookahead window
            tcpa = metrics.get('tcpa', float('inf'))
            dcpa = metrics.get('dcpa', float('inf'))
            
            # Threat if: will be close soon (TCPA < tp) AND will be too close (DCPA < Cs)
            if tcpa <= tp and dcpa <= Cs:
                return True
        except Exception:
            # Fallback: use simple distance if API fails
            dist = np.sqrt((pos_x - ox)**2 + (pos_y - oy)**2)
            if dist <= dsafe:
                return True
    
    return False


def L1_check(pos_x: float, pos_y: float, v1_x: float, v1_y: float, delta: float,
              psi: float = None, alignment_threshold: float = np.pi/60) -> bool:
    """
    L1: Check if ||p(t) - V1|| > delta (not yet reached V1)

    If psi is provided, also checks that heading is aligned with V1 direction.
    This ensures the ship is heading toward V1 when transitioning to S3.

    Args:
        pos_x, pos_y: Current ship position
        v1_x, v1_y: Virtual waypoint V1 position
        delta: Arrival tolerance
        psi: Current heading (optional, for alignment check)
        alignment_threshold: Maximum heading error (default 30°)

    Returns:
        bool: True if not yet reached V1 (or not aligned if psi provided)
    """
    dist_to_v1 = np.sqrt((pos_x - v1_x)**2 + (pos_y - v1_y)**2)

    # Basic distance check
    if dist_to_v1 > delta:
        return True

    # If psi provided, also check heading alignment
    if psi is not None:
        angle_to_v1 = np.arctan2(v1_y - pos_y, v1_x - pos_x)
        heading_error = np.abs(np.arctan2(np.sin(psi - angle_to_v1), np.cos(psi - angle_to_v1)))
        # Return True (not reached) if heading not aligned
        if heading_error > alignment_threshold:
            return True

    return False


def L2_check(pos_x: float, pos_y: float, psi: float, v1_x: float, v1_y: float) -> bool:
    """
    L2: Check if V1 is ahead of ship (within ±2π/3 of heading)

    Uses a wider threshold (±120°) to ensure V1 is truly behind before
    triggering transition to S3. This prevents premature transitions when
    V1 is at a steep starboard angle.

    Args:
        pos_x, pos_y: Current ship position
        psi: Current heading
        v1_x, v1_y: Virtual waypoint V1 position

    Returns:
        bool: True if V1 is ahead (within ±120° of heading)
    """
    angle_to_v1 = np.arctan2(v1_y - pos_y, v1_x - pos_x)
    relative_angle = np.arctan2(np.sin(angle_to_v1 - psi), np.cos(angle_to_v1 - psi))
    # Use ±2π/3 (±120°) instead of ±π/2 (±90°) to prevent premature transitions
    return -2*np.pi/3 < relative_angle < 2*np.pi/3
