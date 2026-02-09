from hybrid_automaton.definition import reset
from hybrid_automaton import RuntimeContext
from colav_automaton.controllers import compute_v1, get_unsafe_set_vertices


@reset
def reset_enter_avoidance(ctx: RuntimeContext) -> RuntimeContext:
    """
    Reset when entering S2 (collision avoidance mode).

    Computes virtual waypoint V1 and pushes it onto the waypoints stack.
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration

    def vertex_provider(pos_x, pos_y, obstacles_list, Cs, psi):
        return get_unsafe_set_vertices(
            pos_x, pos_y, obstacles_list, Cs,
            dsf=cfg['dsafe'], ship_psi=psi, ship_v=cfg['v']
        )

    v1 = compute_v1(
        state[0], state[1], state[2],
        cfg['obstacles'], cfg['Cs'],
        vertex_provider, cfg.get('v1_buffer', 0.0)
    )

    if v1 is not None:
        cfg['waypoints'].append(v1)

    return ctx


@reset
def reset_reach_V1(ctx: RuntimeContext) -> RuntimeContext:
    """
    Reset when transitioning S2 -> S3 (V1 reached or behind).

    Pops V1 from the waypoints stack.
    """
    if len(ctx.configuration.get('waypoints', [])) > 1:
        ctx.configuration['waypoints'].pop()
    return ctx


@reset
def reset_exit_avoidance(ctx: RuntimeContext) -> RuntimeContext:
    """
    Reset when exiting S3 back to S1 (resume waypoint reaching).
    """
    return ctx
