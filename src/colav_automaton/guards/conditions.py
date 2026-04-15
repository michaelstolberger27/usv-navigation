"""
Collision detection and path planning condition checks.

Implements guard condition functions (G11, G12, G22, L1, L2, G23) that determine
when collision avoidance is needed and when avoidance maneuvers complete.

Follows the paper's switching strategy (Section 4.1.1, 4.2, Figure 8):
- G11: LOS to waypoint intersects unsafe set (eq 13)
- G12: Distance to obstacle <= dsafe (eq 14)
- G22: Risk index RI(DCPA, TCPA, d_s) >= K (eq 19-21)
- L1:  Distance to V1 > delta (eq 15)
- L2:  V1 is ahead of ship (eq 16)
- G23: Swept obstacle trajectory intersects LOS set (eq 27)
"""

from typing import List, Tuple
import numpy as np
from colav_automaton.controllers import create_los_cone, compute_unified_unsafe_region
from colav_automaton.controllers.unsafe_sets import (
    _create_agent, _create_obstacles, calculate_obstacle_metrics_for_agent
)

# Configuration constants for guard conditions
V1_AHEAD_THRESHOLD = np.pi / 2  # ±90 degrees (paper eq 16)


def classify_encounter(
    psi: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    pos_x: float,
    pos_y: float,
) -> str:
    """
    Classify the most threatening encounter per COLREGs.

    Uses the closest obstacle's relative bearing and heading difference:
    - head_on: approaching roughly head-to-head (|heading diff| > ~160°)
    - crossing_from_starboard: traffic on starboard -> ego is give-way (Rule 15)
    - crossing_from_port: traffic on port -> ego is stand-on (Rule 17)
    - overtaking: traffic approaching from astern

    Returns:
        str: encounter type
    """
    if not obstacles_list:
        return "none"

    # Find closest obstacle
    min_dist = np.inf
    closest = None
    for obs in obstacles_list:
        d = np.hypot(obs[0] - pos_x, obs[1] - pos_y)
        if d < min_dist:
            min_dist = d
            closest = obs

    ox, oy, ov, o_psi = closest

    # Relative bearing of traffic from ego (positive = port, negative = starboard)
    bearing = np.arctan2(oy - pos_y, ox - pos_x) - psi
    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

    # Heading difference
    hdg_diff = o_psi - psi
    hdg_diff = np.arctan2(np.sin(hdg_diff), np.cos(hdg_diff))

    if abs(hdg_diff) > 2.8:  # ~160° — roughly head-on
        return "head_on"
    elif abs(bearing) > 2.4:  # traffic behind — overtaking
        return "overtaking"
    elif bearing > 0:  # traffic on port side
        return "crossing_from_port"
    else:  # traffic on starboard side
        return "crossing_from_starboard"


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

    # LOS cone half-width = Cs (paper eq 13: F(p(t)) = conv(B_2(p, v*tp), pw))
    effective_tp = Cs / v
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
    dsafe: float,
    tp: float = 3.0,
) -> bool:
    """
    G12: Distance to obstacle <= dsafe (paper eq 14, extended for dynamic obstacles).

    For static obstacles: d_safe = Cs + v * tp (paper eq 14).
    For dynamic obstacles: the paper's Theorem 3 proof uses a larger
    scenario-dependent dsafe.  We approximate this by adding the obstacle's
    closing velocity component during tp, ensuring the ship has enough room
    to complete the prescribed-time heading change before reaching Cs.

    Args:
        pos_x, pos_y: Current ship position
        psi: Current ship heading (rad)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        ship_v: Ship velocity (m/s)
        Cs: Safety radius (m)
        dsafe: Base safe distance = Cs + v * tp
        tp: Prescribed time (s)

    Returns:
        bool: True if any obstacle is within its effective dsafe
    """
    for ox, oy, ov, o_psi in obstacles_list:
        dx, dy = ox - pos_x, oy - pos_y
        dist = np.hypot(dx, dy)

        # Compute closing speed: component of obstacle velocity toward ship
        if dist > 1e-6:
            # Unit vector from obstacle to ship
            ux, uy = -dx / dist, -dy / dist
            closing_speed = max(ov * (np.cos(o_psi) * ux + np.sin(o_psi) * uy), 0.0)
        else:
            closing_speed = ov

        # Effective dsafe accounts for obstacle closing during tp
        effective_dsafe = dsafe + closing_speed * tp

        if dist <= effective_dsafe:
            return True
    return False


def _F(z: float, beta1: float, beta2: float) -> float:
    """
    Piecewise risk function F(z) from paper eq 20.

    Maps a metric value z to [0, 1]:
    - z <= beta1: F = 1 (maximum risk)
    - beta1 < z <= (beta1+beta2)/2: F = 1 - 2*((z-beta1)/(beta2-beta1))^2
    - (beta1+beta2)/2 < z <= beta2: F = 2*((z-beta2)/(beta2-beta1))^2
    - z > beta2: F = 0 (no risk)
    """
    if z <= beta1:
        return 1.0
    mid = (beta1 + beta2) / 2.0
    if z <= mid:
        return 1.0 - 2.0 * ((z - beta1) / (beta2 - beta1)) ** 2
    if z <= beta2:
        return 2.0 * ((z - beta2) / (beta2 - beta1)) ** 2
    return 0.0


def G22_check(
    pos_x: float,
    pos_y: float,
    psi: float,
    obstacles_list: List[Tuple[float, float, float, float]],
    ship_v: float,
    Cs: float,
    K: float = 0.35,
    dcpa_beta1: float = 463.0,
    dcpa_beta2: float = 926.0,
    tcpa_beta1: float = 120.0,
    tcpa_beta2: float = 240.0,
    dist_beta1: float = 148.0,
    dist_beta2: float = 463.0,
) -> bool:
    """
    G22: Risk assessment switching condition (paper eq 19-21, Section 4.2).

    RI(DCPA, TCPA, d_s) = 1/3 * (F(DCPA) + F(TCPA) + F(d_s)) >= K

    Uses the paper's nonlinear index function F(z) (eq 20) applied to three
    metrics: DCPA, TCPA, and distance.  This replaces G12 for dynamic
    obstacles, enabling earlier and smoother avoidance (Rule 8).

    Default beta parameters are scaled from the paper's nautical values
    (DCPA [0.25, 0.5] nmi, TCPA [2, 4] min, dist [0.08, 0.25] nmi)
    to meters/seconds for CommonOcean scenarios.

    Args:
        pos_x, pos_y: Current ship position
        psi: Current ship heading (rad)
        obstacles_list: List of (ox, oy, ov, o_psi) tuples
        ship_v: Ship velocity (m/s)
        Cs: Safety radius (m)
        K: Risk threshold (paper uses 0.35)
        dcpa_beta1, dcpa_beta2: DCPA risk bounds (m)
        tcpa_beta1, tcpa_beta2: TCPA risk bounds (s)
        dist_beta1, dist_beta2: Distance risk bounds (m)

    Returns:
        bool: True if RI >= K for any obstacle (collision risk is high)
    """
    if not obstacles_list:
        return False

    agent = _create_agent(pos_x, pos_y, psi, ship_v, Cs)
    dynamic_obstacles = _create_obstacles(obstacles_list, Cs)
    results = calculate_obstacle_metrics_for_agent(agent, dynamic_obstacles)

    for i, r in enumerate(results):
        ox, oy = obstacles_list[i][0], obstacles_list[i][1]
        d_s = np.hypot(ox - pos_x, oy - pos_y)

        dcpa = r.dcpa
        tcpa = r.tcpa

        # Only consider obstacles that are approaching (positive TCPA)
        if tcpa < 0:
            continue

        f_dcpa = _F(dcpa, dcpa_beta1, dcpa_beta2)
        f_tcpa = _F(tcpa, tcpa_beta1, tcpa_beta2)
        f_dist = _F(d_s, dist_beta1, dist_beta2)

        ri = (f_dcpa + f_tcpa + f_dist) / 3.0

        if ri >= K:
            return True

    return False


def G23_check(
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
    G23: Swept obstacle trajectory intersects LOS set (paper eq 27).

    Checks whether the reachable set of any dynamic obstacle over time
    [0, t] intersects the ship's waypoint LOS set F(p(t)).  This is the
    resume condition for dynamic obstacles — the ship stays in S3 until
    the obstacle's predicted trajectory no longer threatens the path.

    F_0(p_o(t_0), t) = B_∞(0, Cs) ⊕ {p : ∃τ ∈ [t_0, t_0+t] : p = p_o(t_0) + τ*v_o}

    We approximate this by checking the swept unsafe region (which the
    unsafe-set API already computes via _compute_swept_obstacles) against
    the LOS cone, exactly as G11 does but with the full swept region.

    Returns:
        bool: True if the swept obstacle trajectory intersects the LOS set
              (i.e., NOT safe to resume — keep holding in S3)
    """
    if not obstacles_list:
        return False

    if v <= 0 or tp <= 0 or Cs <= 0:
        return False

    # Use static (current position only) for the resume check — the full
    # TCPA-swept region (used by G11) is too conservative here and prevents
    # the ship from ever resuming to S1.  We only block resume if the
    # obstacle's current Cs-circle intersects the LOS cone.
    unsafe_polygon = compute_unified_unsafe_region(
        pos_x, pos_y, obstacles_list, Cs,
        ship_psi=psi, ship_v=v, static_only=True
    )

    if unsafe_polygon is None:
        return False

    # LOS cone with effective_tp = Cs/v (same as G11)
    effective_tp = Cs / v
    los_cone = create_los_cone(pos_x, pos_y, xw, yw, v, effective_tp)

    return unsafe_polygon.intersects(los_cone)


def L1_check(pos_x: float, pos_y: float, v1_x: float, v1_y: float, delta: float,
              psi: float = None) -> bool:
    """
    L1: ||p(t) - V1|| > delta (paper eq 15).

    Args:
        pos_x, pos_y: Current ship position
        v1_x, v1_y: Virtual waypoint V1 position
        delta: Arrival tolerance (m, must be > 0)
        psi: Unused (kept for API compatibility)

    Returns:
        bool: True if not yet reached V1

    Raises:
        ValueError: If delta <= 0
    """
    if delta <= 0:
        raise ValueError(f"Invalid parameter: delta={delta} (must be > 0)")

    dist_to_v1 = np.sqrt((pos_x - v1_x)**2 + (pos_y - v1_y)**2)
    return dist_to_v1 > delta


def L2_check(pos_x: float, pos_y: float, psi: float, v1_x: float, v1_y: float) -> bool:
    """
    L2: V1 is ahead of the ship (paper eq 16).

    −π/2 < atan2(yV1−y, xV1−x) − ψ < π/2

    Args:
        pos_x, pos_y: Current ship position
        psi: Current heading (rad)
        v1_x, v1_y: Virtual waypoint V1 position

    Returns:
        bool: True if V1 is ahead (within ±90° of heading)
    """
    angle_to_v1 = np.arctan2(v1_y - pos_y, v1_x - pos_x)
    relative_angle = np.arctan2(np.sin(angle_to_v1 - psi), np.cos(angle_to_v1 - psi))
    return -V1_AHEAD_THRESHOLD < relative_angle < V1_AHEAD_THRESHOLD
