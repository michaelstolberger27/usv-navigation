from typing import List, Tuple, Optional

import numpy as np

from hybrid_automaton import Automaton
from hybrid_automaton.definition import State, Transition
from colav_automaton.guards import G11_and_G22_guard, L1_bar_or_L2_bar_guard, not_G23_guard
from colav_automaton.resets import reset_enter_avoidance, reset_reach_V1, reset_exit_avoidance
from colav_automaton.invariants import is_goal_waypoint_invariant
from colav_automaton.dynamics import (
    waypoint_navigation_dynamics,
    constant_control_dynamics,
)


def _compute_delta(v: float, a: float, eta: float, tp: float, m: float) -> float:
    """Compute δ from paper eq 9 (with t0 = 0).

    δ ≥ 2v / (a · (m − π − η·π / (a · tp^η)))

    where m = (c − a·π) / a is the input constraint bound (|u| ≤ m).

    When the denominator is non-positive, the input constraint |u| ≤ m is
    satisfied for ANY δ > 0 (the bound is non-binding).  This is the common
    case for typical parameters (e.g. m=3, η=3.5, a=1.67) — a practical
    heuristic is used instead.
    """
    denom_inner = m - np.pi - eta * np.pi / (a * tp ** eta)
    if denom_inner <= 0:
        # Non-binding: any δ > 0 satisfies the input constraint.
        # Use a practical heuristic for the waypoint arrival radius.
        return max(5.0, v * tp * 0.5)
    denom = a * denom_inner
    delta = 2.0 * v / denom
    return max(delta, 1.0)


def ColavAutomaton(
    waypoint_x: float = 10.0,
    waypoint_y: float = 9.0,
    obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
    Cs: float = 2.0,
    a: float = 1.67,
    v: float = 12.0,
    eta: float = 3.5,
    tp: float = 1.0,
    v1_buffer: float = 0.0,
    m: float = 3.0,
    K: float = 0.35,
    dcpa_beta1: float = 463.0,
    dcpa_beta2: float = 926.0,
    tcpa_beta1: float = 120.0,
    tcpa_beta2: float = 240.0,
    dist_beta1: float = 148.0,
    dist_beta2: float = 463.0,
) -> Automaton:

    delta = _compute_delta(v, a, eta, tp, m)
    dsafe = Cs + v * tp

    # States
    S1 = State(
        name="WAYPOINT_REACHING",
        initial=True,
        flow=waypoint_navigation_dynamics,
        invariants=[is_goal_waypoint_invariant],
        on_enter=lambda: print("S1: WAYPOINT_REACHING")
    )

    S2 = State(
        name="COLLISION_AVOIDANCE",
        flow=waypoint_navigation_dynamics,
        on_enter=lambda: print("S2: COLLISION_AVOIDANCE")
    )

    S3 = State(
        name="CONSTANT_CONTROL",
        flow=constant_control_dynamics,
        on_enter=lambda: print("S3: CONSTANT_CONTROL")
    )

    # Transitions
    S1.add_transition(Transition(
        name="avoid",
        to_state=S2,
        guards=[G11_and_G22_guard],
        reset=reset_enter_avoidance,
        priority=1
    ))

    S2.add_transition(Transition(
        name="hold",
        to_state=S3,
        guards=[L1_bar_or_L2_bar_guard],
        reset=reset_reach_V1,
        priority=1
    ))

    S3.add_transition(Transition(
        name="resume",
        to_state=S1,
        guards=[not_G23_guard],
        reset=reset_exit_avoidance,
        priority=1
    ))

    ha = Automaton(
        name="COLAV Automaton",
        version="0.0.1",
        states=[S1, S2, S3],
        configuration={
            'waypoints': [(waypoint_x, waypoint_y)],
            'obstacles': obstacles or [],
            'Cs': Cs,
            'dsafe': dsafe,
            'delta': delta,
            'a': a,
            'v': v,
            'eta': eta,
            'tp': tp,
            'v1_buffer': v1_buffer,
            'K': K,
            'dcpa_beta1': dcpa_beta1,
            'dcpa_beta2': dcpa_beta2,
            'tcpa_beta1': tcpa_beta1,
            'tcpa_beta2': tcpa_beta2,
            'dist_beta1': dist_beta1,
            'dist_beta2': dist_beta2,
        }
    )

    return ha
