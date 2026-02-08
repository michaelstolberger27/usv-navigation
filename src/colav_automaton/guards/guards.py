from hybrid_automaton import guard
from hybrid_automaton.automaton_runtime_context import Context
from colav_automaton.guards.conditions import (
    G11_check, G12_check, L1_check, L2_check
)


@guard
def G11_and_G12_guard(ctx: Context) -> bool:
    """
    Guard for S1 -> S2 transition: Enter collision avoidance.

    Activates when obstacle is in path (G11) AND close enough (G12).
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    G12 = G12_check(
        state[0], state[1], state[2], cfg['obstacles'],
        cfg['v'], cfg['Cs'], cfg['dsafe']
    )

    return G11 and G12


@guard
def L1_bar_or_L2_bar_guard(ctx: Context) -> bool:
    """
    Guard for S2 -> S3 transition: Enter constant control.

    Activates when V1 reached (not L1) OR V1 is behind (not L2).
    V1 is the top of the waypoints stack (pushed by reset_enter_avoidance).
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    if len(cfg.get('waypoints', [])) < 2:
        return False

    v1_x, v1_y = cfg['waypoints'][-1]

    L1 = L1_check(state[0], state[1], v1_x, v1_y, cfg['delta'], psi=state[2])
    L2 = L2_check(state[0], state[1], state[2], v1_x, v1_y)

    return (not L1) or (not L2)


@guard
def not_G11_guard(ctx: Context) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching.

    Activates when LOS to current target waypoint is clear (not G11).
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    target_x, target_y = cfg['waypoints'][-1]

    G11 = G11_check(
        state[0], state[1], state[2],
        target_x, target_y,
        cfg['v'], cfg['tp'],
        cfg['obstacles'], cfg['Cs']
    )

    return not G11
