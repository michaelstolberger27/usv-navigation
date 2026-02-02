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
