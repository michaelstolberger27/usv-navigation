"""
Deterministic tick-synchronous executive for the COLAV automaton.

Same formal automaton as ColavAutomaton — identical guards, resets,
dynamics, V1 logic, and configuration — but stepped synchronously:
one step(dt) call evaluates control, integrates the flow, and fires at
most one transition, with the prescribed-time clock driven by *sim*
time rather than wall time. No threads, no asyncio, no sleeps: given
the same inputs, two runs produce bit-identical trajectories.

This mirrors hybrid_automaton's per-step semantics (integrate first,
then guards, highest-priority transition wins, transition resets the
clock) while replacing its wall-clock scheduling. The asynchronous
runtime samples guards and control at unpredictable wall-clock
instants, which makes outcomes load-dependent (HANDOFF §3) and is a
*different* discretisation of the paper's model on every run; the
fixed-dt evaluation here is the known, analyzable one.

Architecture note: asynchronous I/O (AIS listeners, ROS topics, UIs)
belongs at the edges, writing into buffers; whoever owns the clock —
a sim loop, or a real-time timer on the vessel — calls step() with a
snapshot. The control core itself stays deterministic.

Usage::

    rt = SyncColavRuntime(
        waypoint=(4000.0, 0.0),
        obstacles=[(1500.0, 80.0, 5.0, np.pi)],
        initial_state=(0.0, 0.0, 0.0),
        Cs=300.0, v=6.0, tp=3.0,
    )
    while not rt.goal_reached():
        result = rt.step(dt=1.0, obstacles=current_obstacles())
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from colav_automaton.automaton import _compute_delta
from colav_automaton.controllers.prescribed_time import (
    compute_prescribed_time_control,
)
from colav_automaton.dynamics import (
    constant_control_flow,
    waypoint_navigation_flow,
)
from colav_automaton.guards import (
    G11_and_G22_guard,
    L1_bar_or_L2_bar_guard,
    not_G23_guard,
)
from colav_automaton.resets import (
    apply_enter_avoidance,
    apply_exit_avoidance,
    apply_reach_V1,
)

S1 = "WAYPOINT_REACHING"
S2 = "COLLISION_AVOIDANCE"
S3 = "CONSTANT_CONTROL"


class _Buffer:
    """Minimal stand-in for the runtime's buffered value wrappers."""

    def __init__(self, value: np.ndarray):
        self._value = np.asarray(value, dtype=float)

    def latest(self) -> np.ndarray:
        return self._value

    def add(self, value: np.ndarray) -> None:
        self._value = np.asarray(value, dtype=float)

    # API-compat alias used by the async runtime's state wrapper
    def set_continuous_state(self, value: np.ndarray) -> None:
        self.add(value)


class _ScalarBuffer:
    """
    Buffer for the scalar control input u. Stores a plain float so the
    shared code's float(buffer.latest()) is exact and warning-free
    (float() of a shape-(1,) array is deprecated in numpy >= 1.25).
    """

    def __init__(self, value: float):
        self._value = float(np.ravel(value)[0]) if np.ndim(value) else float(value)

    def latest(self) -> float:
        return self._value

    def add(self, value) -> None:
        self._value = float(np.ravel(value)[0]) if np.ndim(value) else float(value)


class _SyncContext:
    """
    Duck-typed RuntimeContext exposing exactly what the decorated
    guard/reset/dynamics functions read (they are invoked via .func,
    bypassing the framework's isinstance check):
    continuous_state.latest(), configuration, control_input_states['u'].
    """

    def __init__(self, state: np.ndarray, configuration: dict, u0: float):
        self.continuous_state = _Buffer(state)
        self.configuration = configuration
        self.control_input_states = {'u': _ScalarBuffer(u0)}


@dataclass
class StepResult:
    t: float                      # sim time after the step
    state: np.ndarray             # [x, y, psi] after the step
    mode: str                     # discrete state after the step
    u: float                      # control input used this step
    transition: Optional[str]     # transition name if one fired, else None


# (name, target mode, guard, reset) per source mode — wiring identical
# to ColavAutomaton's Transition declarations. Guards are @guard
# annotations (called via .func); resets and flows are the plain shared
# implementations.
_TRANSITIONS = {
    S1: [("avoid", S2, G11_and_G22_guard, apply_enter_avoidance)],
    S2: [("hold", S3, L1_bar_or_L2_bar_guard, apply_reach_V1)],
    S3: [("resume", S1, not_G23_guard, apply_exit_avoidance)],
}

_DYNAMICS = {
    S1: waypoint_navigation_flow,
    S2: waypoint_navigation_flow,
    S3: constant_control_flow,
}


class SyncColavRuntime:
    """Tick-synchronous COLAV automaton (see module docstring)."""

    def __init__(
        self,
        waypoint: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
        initial_state: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        Cs: float = 2.0,
        a: float = 1.67,
        v: float = 12.0,
        eta: float = 3.5,
        tp: float = 1.0,
        v1_buffer: float = 0.0,
        m: float = 3.0,
        K: float = 0.35,
        K_off: float = 0.25,
        dcpa_beta1: float = 463.0,
        dcpa_beta2: float = 926.0,
        tcpa_beta1: float = 120.0,
        tcpa_beta2: float = 240.0,
        dist_beta1: float = 148.0,
        dist_beta2: float = 463.0,
    ):
        # Configuration mirrors ColavAutomaton exactly so the guard/
        # reset/dynamics functions see the same keys either way.
        configuration = {
            'waypoints': [tuple(waypoint)],
            'obstacles': list(obstacles or []),
            'Cs': Cs,
            'dsafe': Cs + v * tp,                       # paper eq 14
            'delta': _compute_delta(v, a, eta, tp, m),  # paper eq 9
            'a': a,
            'v': v,
            'eta': eta,
            'tp': tp,
            'v1_buffer': v1_buffer,
            'K': K,
            'K_off': K_off,
            'dcpa_beta1': dcpa_beta1,
            'dcpa_beta2': dcpa_beta2,
            'tcpa_beta1': tcpa_beta1,
            'tcpa_beta2': tcpa_beta2,
            'dist_beta1': dist_beta1,
            'dist_beta2': dist_beta2,
        }
        psi0 = float(initial_state[2])
        self._ctx = _SyncContext(np.array(initial_state, dtype=float),
                                 configuration, u0=psi0)
        self.mode = S1
        self.t = 0.0
        self._t_last_transition = 0.0

    # ---- public accessors ----

    @property
    def state(self) -> np.ndarray:
        return self._ctx.continuous_state.latest().copy()

    @property
    def configuration(self) -> dict:
        return self._ctx.configuration

    @property
    def current_waypoint(self) -> Tuple[float, float]:
        return tuple(self._ctx.configuration['waypoints'][-1])

    def goal_reached(self, radius: Optional[float] = None) -> bool:
        gx, gy = self._ctx.configuration['waypoints'][0]
        x, y, _ = self._ctx.continuous_state.latest()
        r = radius if radius is not None else self._ctx.configuration['delta']
        return bool(np.hypot(gx - x, gy - y) < r)

    def notify_waypoint_changed(self) -> None:
        """
        Reset the prescribed-time clock after an external waypoint
        change (the async adapter pings the clock the same way).
        """
        self._t_last_transition = self.t

    # ---- the tick ----

    def step(
        self,
        dt: float,
        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> StepResult:
        """
        Advance the automaton by one tick of sim time.

        Order mirrors hybrid_automaton's _evaluation_step: control,
        flow integration, then guards with at most one (highest-
        priority) transition whose reset runs immediately and resets
        the prescribed-time clock.
        """
        ctx = self._ctx
        if obstacles is not None:
            ctx.configuration['obstacles'] = list(obstacles)

        # 1. Control: prescribed-time heading law toward the active
        #    waypoint, with t measured in sim time since the last
        #    transition (the async runtime measures wall time — that is
        #    the nondeterminism this class exists to remove).
        state = ctx.continuous_state.latest()
        wx, wy = ctx.configuration['waypoints'][-1]
        cfg = ctx.configuration
        u = compute_prescribed_time_control(
            self.t - self._t_last_transition,
            state[0], state[1], state[2], wx, wy,
            a=cfg['a'], v=cfg['v'], eta=cfg['eta'], tp=cfg['tp'],
        )
        ctx.control_input_states['u'].add(np.array([u]))

        # 2. Continuous flow (Euler, like the async runtime's integrate)
        xdot = _DYNAMICS[self.mode](ctx)
        new_state = ctx.continuous_state.latest() + xdot * dt
        ctx.continuous_state.add(new_state)
        self.t += dt

        # 3. Guards — at most one transition per tick
        fired = None
        for name, target, guard, reset_fn in _TRANSITIONS[self.mode]:
            if guard.func(ctx):
                reset_fn(ctx)
                self.mode = target
                self._t_last_transition = self.t
                fired = name
                break

        return StepResult(
            t=self.t,
            state=ctx.continuous_state.latest().copy(),
            mode=self.mode,
            u=float(u),
            transition=fired,
        )
