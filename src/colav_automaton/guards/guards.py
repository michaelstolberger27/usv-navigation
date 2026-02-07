import numpy as np
from hybrid_automaton import guard
from hybrid_automaton.automaton_runtime_context import Context
from colav_automaton.guards.conditions import (
    check_G11_dynamic, check_G12_dynamic, L1_check, L2_check
)

# ============================================================================
# High-Level Guard Functions
# ============================================================================

@guard
def G11_and_G12_guard(ctx: Context) -> bool:
    """
    Guard for S1 -> S2 transition: Enter collision avoidance

    Activates when obstacle is in path (G11) AND close enough (G12).
    Uses the unsafe-set API for dynamic obstacle computation.
    Checks LOS to current target waypoint (top of stack).

    Args:
        ctx: Context containing continuous state [x, y, psi], auxiliary states,
             control inputs, configuration (waypoints, obstacle(s), v, tp, Cs, dsafe),
             and clock

    Returns:
        bool: True if should enter collision avoidance
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    # Get current target waypoint from stack (top of stack = LIFO)
    target_x, target_y = cfg['waypoints'][-1]

    G11 = check_G11_dynamic(
        state[0], state[1],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'],
        cfg['Cs']
    )

    G12 = check_G12_dynamic(
        state[0], state[1], cfg['obstacles'], cfg['dsafe'],
        ship_v=cfg['v'], tp=cfg['tp'], Cs=cfg['Cs']
    )

    return G11 and G12


@guard
def L1_bar_or_L2_bar_guard(ctx: Context) -> bool:
    """
    Guard for S2 -> S3 transition: Enter constant control

    Activates when V1 reached (L1) OR V1 is behind (L2)

    Docs:
        Corresponds to guard L1 ∨ L2 in ship navigation automaton

    Args:
        ctx: Context containing continuous state [x, y, psi], auxiliary states,
             control inputs, configuration (delta, ca_controller), and clock

    Returns:
        bool: True if should enter constant control mode
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    
    # Need virtual waypoint from controller (set by dynamics)
    if 'ca_controller' not in cfg or cfg['ca_controller'] is None:
        return False
    
    if cfg['ca_controller'].virtual_waypoint is None:
        return False
    
    v1_x, v1_y = cfg['ca_controller'].virtual_waypoint

    # L1 check includes heading alignment to ensure ship is pointing at V1
    L1 = L1_check(state[0], state[1], v1_x, v1_y, cfg['delta'], psi=state[2])
    L2 = L2_check(state[0], state[1], state[2], v1_x, v1_y)

    return (not L1) or (not L2)


@guard
def not_G11_guard(ctx: Context) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching

    Activates when LOS to current target waypoint is clear (not G11).
    Uses the unsafe-set API for dynamic obstacle computation.
    Checks LOS to current target waypoint (top of stack).

    Args:
        ctx: Context containing continuous state [x, y, psi], auxiliary states,
             control inputs, configuration (waypoints, obstacle(s), v, tp, Cs),
             and clock

    Returns:
        bool: True if LOS is clear and can resume waypoint reaching
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    # Get current target waypoint from stack (top of stack = LIFO)
    target_x, target_y = cfg['waypoints'][-1]

    G11 = check_G11_dynamic(
        state[0], state[1],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'],
        cfg['Cs']
    )

    return not G11