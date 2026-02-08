import numpy as np
from hybrid_automaton.automaton_annotations import continuous_dynamics
from hybrid_automaton.automaton_runtime_context import Context


@continuous_dynamics
def waypoint_navigation_dynamics(ctx: Context):
    """
    Shared dynamics for S1 and S2: navigate to top of waypoints stack.

    In S1 the target is the goal waypoint. In S2 the target is V1
    (pushed onto the stack by the S1->S2 reset).

    Uses prescribed-time control to guarantee heading convergence to LOS.
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    t = ctx.clk.get_time_elapsed_since_transition()

    target_x, target_y = cfg['waypoints'][-1]

    return cfg['pt_controller'].compute_dynamics(
        t, state[0], state[1], state[2],
        target_x, target_y
    )


@continuous_dynamics
def constant_control_dynamics(ctx: Context):
    """
    S3: Constant heading - maintain straight-line motion after avoidance.

    Holds the current heading with zero turning rate until LOS is clear.
    """
    v = ctx.cfg['v']
    psi = ctx.x.latest()[2]

    return np.array([v * np.cos(psi), v * np.sin(psi), 0.0])
