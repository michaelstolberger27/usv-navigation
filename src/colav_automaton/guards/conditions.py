"""
Collision detection and path planning condition checks.

Implements guard condition functions (G11, G12, L1, L2) that determine
when collision avoidance is needed and when avoidance maneuvers complete.
"""

from typing import List, Tuple
import numpy as np
from colav_controllers import create_los_cone, compute_unified_unsafe_region, check_collision_threat

# Configuration constants for guard conditions
HEADING_ALIGNMENT_THRESHOLD = np.pi / 60  # ~3 degrees
V1_AHEAD_THRESHOLD = 2 * np.pi / 3  # ±120 degrees - prevent premature S2->S3 transitions


def G11_check(
    pos_x: float,
    pos_y: float,
    psi: float,
    xw: float,
    yw: float,
    v: float,
    tp: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    Cs: float
) -> bool:
    """
    G11: Check if LOS to waypoint intersects any unsafe region.

    Uses swept unsafe regions to account for obstacle motion.

    Args:
        pos_x, pos_y: Current ship position
        psi: Current ship heading (rad)
        xw, yw: Waypoint position
        v: Ship velocity (m/s, must be > 0)
        tp: Prescribed time (s, must be > 0)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        Cs: Safety radius for obstacles (m, must be > 0)

    Returns:
        bool: True if LOS cone intersects any unsafe region
        
    Raises:
        ValueError: If any velocity or time parameter is invalid
    """
    if not obstacles_list:
        return False

    if v <= 0 or tp <= 0 or Cs <= 0:
        raise ValueError(f"Invalid parameters: v={v}, tp={tp}, Cs={Cs} (all must be > 0)")

    unsafe_polygon = compute_unified_unsafe_region(
        pos_x, pos_y, obstacles_list, Cs,
        ship_psi=psi, ship_v=v
    )

    if unsafe_polygon is None:
        return False

    # Use a narrower LOS cone for checking (reduced uncertainty after maneuvers)
    # This prevents false positives when ship has already passed obstacle
    effective_tp = min(tp, 1.0)  # Cap at 1 second of travel to prevent excessively wide cones
    los_cone = create_los_cone(pos_x, pos_y, xw, yw, v, effective_tp)

    # Check intersection
    return unsafe_polygon.intersects(los_cone)


def G12_check(
    pos_x: float,
    pos_y: float,
    psi: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    ship_v: float,
    Cs: float,
    dsafe: float
) -> bool:
    """
    G12: Check if any obstacle poses a collision threat using DCPA/TCPA.

    An obstacle threatens if:
    - Time to Closest Point of Approach (TCPA) <= dsafe / ship_v
    - Distance at CPA (DCPA) <= safe maneuvering distance (dsafe)

    The TCPA threshold is derived from the safe distance and ship speed,
    giving enough lead time for the prescribed-time controller to maneuver.

    Args:
        pos_x, pos_y: Current ship position
        psi: Current ship heading (rad)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        ship_v: Ship velocity (m/s, must be > 0)
        Cs: Safety radius (m, must be > 0)
        dsafe: Safe maneuvering distance (m, must be > 0)

    Returns:
        bool: True if any obstacle is a collision threat
        
    Raises:
        ValueError: If any velocity or distance parameter is invalid
    """
    if ship_v <= 0 or Cs <= 0 or dsafe <= 0:
        raise ValueError(f"Invalid parameters: ship_v={ship_v}, Cs={Cs}, dsafe={dsafe} (all must be > 0)")

    return check_collision_threat(pos_x, pos_y, psi, obstacles_list, ship_v, Cs, dsafe)


def L1_check(pos_x: float, pos_y: float, v1_x: float, v1_y: float, delta: float,
              psi: float = None) -> bool:
    """
    L1: Check if ||p(t) - V1|| > delta (not yet reached V1)

    If psi is provided, also checks that heading is aligned with V1 direction.
    This ensures the ship is heading toward V1 when transitioning to S3.

    Args:
        pos_x, pos_y: Current ship position
        v1_x, v1_y: Virtual waypoint V1 position
        delta: Arrival tolerance (m, must be > 0)
        psi: Current heading in radians (optional, for alignment check)

    Returns:
        bool: True if not yet reached V1 (or not aligned if psi provided)
        
    Raises:
        ValueError: If delta <= 0
    """
    if delta <= 0:
        raise ValueError(f"Invalid parameter: delta={delta} (must be > 0)")
    
    dist_to_v1 = np.sqrt((pos_x - v1_x)**2 + (pos_y - v1_y)**2)

    # Basic distance check
    if dist_to_v1 > delta:
        return True

    # If psi provided, also check heading alignment
    if psi is not None:
        angle_to_v1 = np.arctan2(v1_y - pos_y, v1_x - pos_x)
        heading_error = np.abs(np.arctan2(np.sin(psi - angle_to_v1), np.cos(psi - angle_to_v1)))
        # Return True (not reached) if heading not aligned
        if heading_error > HEADING_ALIGNMENT_THRESHOLD:
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
        psi: Current heading (rad)
        v1_x, v1_y: Virtual waypoint V1 position

    Returns:
        bool: True if V1 is ahead (within ±120° of heading)
    """
    angle_to_v1 = np.arctan2(v1_y - pos_y, v1_x - pos_x)
    relative_angle = np.arctan2(np.sin(angle_to_v1 - psi), np.cos(angle_to_v1 - psi))
    # Use V1_AHEAD_THRESHOLD (±120°) to prevent premature S2->S3 transitions
    return -V1_AHEAD_THRESHOLD < relative_angle < V1_AHEAD_THRESHOLD
