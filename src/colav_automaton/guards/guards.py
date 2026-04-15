import numpy as np
from hybrid_automaton.definition import guard
from hybrid_automaton import RuntimeContext
from colav_automaton.guards.conditions import (
    G11_check, G12_check, G22_check, G23_check, L1_check, L2_check
)


@guard
def G11_and_G12_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S1 -> S2 transition: Enter collision avoidance.

    Activates when obstacle is in path (G11) AND close enough (G12).
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    G12 = G12_check(
        state[0], state[1], state[2], cfg['obstacles'],
        cfg['v'], cfg['Cs'], cfg['dsafe'], cfg['tp']
    )

    return G11 and G12


@guard
def G11_and_G22_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S1 -> S2 transition: Enter collision avoidance.

    Paper Section 4.2: G11 ∧ G22.  G11 ensures the obstacle's unsafe set
    intersects the LOS cone (obstacle is in the path and close enough).
    G22 ensures the risk index is high enough to warrant avoidance.

    G11 uses static (current position) unsafe regions.  This means crossing
    encounters at wide angles may trigger later than ideal, but it ensures
    V1 is always at a reachable distance (obstacle is nearby when S2 starts).
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    if not G11:
        return False

    return G22_check(
        state[0], state[1], state[2], cfg['obstacles'],
        cfg['v'], cfg['Cs'],
        K=cfg.get('K', 0.35),
        dcpa_beta1=cfg.get('dcpa_beta1', 463.0),
        dcpa_beta2=cfg.get('dcpa_beta2', 926.0),
        tcpa_beta1=cfg.get('tcpa_beta1', 120.0),
        tcpa_beta2=cfg.get('tcpa_beta2', 240.0),
        dist_beta1=cfg.get('dist_beta1', 148.0),
        dist_beta2=cfg.get('dist_beta2', 463.0),
    )


@guard
def L1_bar_or_L2_bar_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S2 -> S3 transition: Enter constant control.

    Activates when V1 reached (not L1) OR V1 is behind (not L2).
    V1 is the top of the waypoints stack (pushed by reset_enter_avoidance).
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    if len(cfg.get('waypoints', [])) < 2:
        return False

    v1_x, v1_y = cfg['waypoints'][-1]

    L1 = L1_check(state[0], state[1], v1_x, v1_y, cfg['delta'], psi=state[2])
    L2 = L2_check(state[0], state[1], state[2], v1_x, v1_y)

    return (not L1) or (not L2)


@guard
def not_G11_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching.

    Paper Figure 8: activates when Ḡ11 (LOS to waypoint is clear).
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    return not G11


@guard
def not_G11_and_not_G12_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching.

    Requires BOTH:
    - ¬G11: LOS to waypoint is clear (no obstacle in path)
    - ¬G12: No obstacle within dsafe

    This prevents premature resumption while the obstacle is still nearby.
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    if G11:
        return False  # LOS still blocked

    G12 = G12_check(
        state[0], state[1], state[2], cfg['obstacles'],
        cfg['v'], cfg['Cs'], cfg['dsafe'], cfg['tp']
    )

    return not G12


@guard
def not_G23_guard(ctx: RuntimeContext) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching (paper eq 27).

    Uses G23 which checks whether the swept obstacle trajectory still
    intersects the ship's waypoint LOS set.  The ship resumes to S1
    only when ¬G23: the predicted obstacle path no longer threatens
    the route to the waypoint.

    This is the paper's prescribed resume condition for dynamic obstacles,
    replacing the simpler ¬G11 which only checks the current instant.
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    target_x, target_y = cfg['waypoints'][-1]

    g23 = G23_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    return not g23
