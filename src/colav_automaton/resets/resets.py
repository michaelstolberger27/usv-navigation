from hybrid_automaton import reset
from hybrid_automaton.automaton_runtime_context import Context


@reset
def reset_enter_avoidance(ctx: Context) -> Context:
    """
    Reset when entering S2 (collision avoidance mode).

    Clears any previous collision avoidance state to ensure fresh
    computation of virtual waypoint V1.

    Args:
        ctx: Context containing continuous state, auxiliary states,
             control inputs, configuration (ca_controller), and clock

    Returns:
        Context: Unchanged context
    """
    if 'ca_controller' in ctx.cfg and ctx.cfg['ca_controller'] is not None:
        ctx.cfg['ca_controller'].reset()
    return ctx


@reset
def reset_reach_V1(ctx: Context) -> Context:
    """
    Reset when transitioning S2 -> S3 (V1 reached or behind).

    Pops the virtual waypoint V1 from the waypoints stack since it has been
    reached or passed. The next target on the stack becomes active.

    Args:
        ctx: Context containing continuous state, auxiliary states,
             control inputs, configuration (ca_controller, waypoints), and clock

    Returns:
        Context: Unchanged context
    """
    # Pop V1 from waypoints stack (keeps at least goal waypoint)
    if len(ctx.cfg.get('waypoints', [])) > 1:
        ctx.cfg['waypoints'].pop()
    return ctx


@reset
def reset_exit_avoidance(ctx: Context) -> Context:
    """
    Reset when exiting S3 back to S1 (resume waypoint reaching).

    Clears collision avoidance state to prepare for normal waypoint navigation.

    Args:
        ctx: Context containing continuous state, auxiliary states,
             control inputs, configuration (ca_controller), and clock

    Returns:
        Context: Unchanged context
    """
    if 'ca_controller' in ctx.cfg and ctx.cfg['ca_controller'] is not None:
        ctx.cfg['ca_controller'].reset()
    return ctx
