import numpy as np
from hybrid_automaton.definition import reset
from hybrid_automaton import RuntimeContext
from colav_automaton.controllers import compute_v1, get_unsafe_set_vertices, default_vertex_provider


@reset
def reset_enter_avoidance(ctx: RuntimeContext) -> RuntimeContext:
    """
    Reset when entering S2 (collision avoidance mode).

    Computes virtual waypoint V1 from the unsafe set vertices (paper
    Section 4.1 for static, 4.3 for dynamic obstacles).  V1 is the
    vertex of the unsafe set that yields the largest predicted CPA,
    evaluated for both starboard and port directions (rule 17b extension).

    Falls back to simple Cs-circle vertices if the unsafe-set API fails.
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration
    pos_x, pos_y, psi = state[0], state[1], state[2]
    v = cfg['v']
    Cs = cfg['Cs']
    dsafe = cfg['dsafe']

    # Inflate Cs for V1 vertex computation to add clearance margin.
    # The ship's curved trajectory during the heading change erodes margin,
    # so V1 should be further from the obstacle than the bare Cs boundary.
    # Default margin = 0.25 * Cs balances CPA improvement vs collision risk.
    v1_Cs_margin = cfg.get('v1_Cs_margin', 0.25 * Cs)
    v1_Cs = Cs + v1_Cs_margin

    def vertex_provider(px, py, obstacles_list, cs, heading):
        # V1 is computed from the obstacle's current unsafe set (no swept
        # region) — the ship steers around where the obstacle IS now.
        # The swept region would place V1 far along the obstacle's trajectory,
        # making it unreachable.
        vertices = get_unsafe_set_vertices(
            px, py, obstacles_list, v1_Cs,
            dsf=dsafe, ship_psi=heading, ship_v=v,
            use_swept_region=False
        )
        if vertices is not None:
            return vertices
        return default_vertex_provider(px, py, obstacles_list, v1_Cs, heading)

    v1 = compute_v1(
        pos_x, pos_y, psi,
        cfg['obstacles'], v1_Cs,
        vertex_provider, cfg.get('v1_buffer', 0.0),
        v=v,
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

    Crucially, resets the control input u to the current heading ψ.
    During S3, the HeadingControlProvider continues computing u with
    t_elapsed approaching tp, which can make the prescribed-time term
    η·e/(a·(tp−t)) near-singular.  S3 dynamics (ψ̇=0) ignore this,
    but the stale u stays in the buffer.  Without this reset, S1's
    first dynamics step would read that enormous u and produce an
    instantaneous heading jump.  Setting u=ψ gives ψ̇=−aψ+aψ=0
    for the first step, allowing the provider to compute a proper
    value on the next tick.
    """
    state = ctx.continuous_state.latest()
    psi = state[2]
    ctx.control_input_states['u'].add(np.array([psi]))
    return ctx
