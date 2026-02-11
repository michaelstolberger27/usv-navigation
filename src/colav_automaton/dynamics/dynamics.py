import numpy as np
from hybrid_automaton.definition import continuous_dynamics
from hybrid_automaton import RuntimeContext


@continuous_dynamics
def waypoint_navigation_dynamics(ctx: RuntimeContext):
    """
    Vessel dynamics for S1 and S2.

    Reads the control input u from the control input states (computed
    asynchronously by the prescribed-time heading controller) and returns
    the continuous dynamics of the vessel:

        dx/dt   = v * cos(psi)
        dy/dt   = v * sin(psi)
        dpsi/dt = -a * psi + a * u
    """
    state = ctx.continuous_state.latest()
    cfg = ctx.configuration
    psi = state[2]
    a = cfg['a']
    v = cfg['v']
    u = float(ctx.control_input_states['u'].latest())

    return np.array([
        v * np.cos(psi),
        v * np.sin(psi),
        -a * psi + a * u
    ])


@continuous_dynamics
def constant_control_dynamics(ctx: RuntimeContext):
    """
    S3: Constant heading - maintain straight-line motion after avoidance.

    Holds the current heading with zero turning rate until LOS is clear.
        dx/dt   = v * cos(psi)
        dy/dt   = v * sin(psi)
        dpsi/dt = 0
    """
    v = ctx.configuration['v']
    psi = ctx.continuous_state.latest()[2]

    return np.array([
        v * np.cos(psi), 
        v * np.sin(psi), 
        0.0
    ])
