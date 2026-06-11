import numpy as np
from hybrid_automaton.definition import continuous_dynamics

# The flow implementations are plain functions so that both runtimes
# share them: the async framework gets decorated wrappers (which
# type-check the framework Context), while SyncColavRuntime calls the
# plain versions with its own duck-typed context.


def waypoint_navigation_flow(ctx):
    """
    Vessel dynamics for S1 and S2.

    Reads the control input u from the control input states (computed
    by the prescribed-time heading controller) and returns the
    continuous dynamics of the vessel:

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


def constant_control_flow(ctx):
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


# Async-framework wrappers (what ColavAutomaton's States use)
waypoint_navigation_dynamics = continuous_dynamics(waypoint_navigation_flow)
constant_control_dynamics = continuous_dynamics(constant_control_flow)
