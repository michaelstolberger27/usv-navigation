"""
Prescribed-time heading controller for ship navigation.

Guarantees heading convergence to line-of-sight (LOS) in prescribed time tp.
"""

import numpy as np


def compute_prescribed_time_control(
    t: float, x: float, y: float, psi: float, xw: float, yw: float,
    *, a: float, v: float, eta: float, tp: float,
) -> float:
    """
    Prescribed-time heading control law.

    Computes the control input u that guarantees heading convergence to
    line-of-sight (LOS) within the prescribed time tp.

        u = (1/a) * dot{psi}_dg + psi - eta * e / (a * (tp - t))   for t < tp
        u = (1/a) * dot{psi}_dg + psi                               for t >= tp

    Args:
        t: Time elapsed since controller activation
        x, y: Current ship position
        psi: Current heading (rad)
        xw, yw: Waypoint position
        a: Heading dynamics coefficient (plant parameter)
        v: Ship velocity (m/s)
        eta: Controller gain (eta > 1)
        tp: Prescribed convergence time (seconds)

    Returns:
        Control input u
    """
    # Desired heading (LOS angle)
    psi_dg = np.arctan2(yw - y, xw - x)

    # Time derivative of desired heading
    dx = xw - x
    dy = yw - y
    d_squared = dx**2 + dy**2

    if d_squared < 1e-6:
        psi_dg_dot = 0.0
    else:
        psi_dg_dot = (-v * dx * np.sin(psi) + v * dy * np.cos(psi)) / d_squared

    # Heading error (wrapped to [-pi, pi])
    e = np.arctan2(np.sin(psi - psi_dg), np.cos(psi - psi_dg))

    # Feedforward + prescribed-time feedback
    if t < tp:
        u = (1 / a) * psi_dg_dot + psi - eta * e / (a * (tp - t + 1e-6))
    else:
        u = (1 / a) * psi_dg_dot + psi

    return u


class HeadingControlProvider:
    """
    Prescribed-time heading controller as a hybrid automaton control provider.

    Runs asynchronously to the automaton evaluation loop, reading the current
    continuous state and computing the control input u for the vessel heading
    dynamics.

    Usage::

        ha = ColavAutomaton(...)
        provider = HeadingControlProvider(ha)

        await ha.activate(
            ...,
            initial_control_input_states={'u': np.array([0.0])},
            control_states_provider=provider,
            control_states_provision_rate=100,
        )
    """

    def __init__(self, automaton):
        self._automaton = automaton

    def __call__(self):
        ctx = self._automaton._runtime._ctx
        cfg = ctx.configuration

        state = ctx.continuous_state.latest()
        t = ctx.clock.get_time_elapsed_since_transition()
        target_x, target_y = cfg['waypoints'][-1]

        u = compute_prescribed_time_control(
            t, state[0], state[1], state[2], target_x, target_y,
            a=cfg['a'], v=cfg['v'], eta=cfg['eta'], tp=cfg['tp'],
        )

        # Update the ControlInput buffer directly (preserves the wrapper objects)
        ctx.control_input_states['u'].add(np.array([u]))

        # Return the existing dict so the framework's inject is a no-op
        return ctx.control_input_states
