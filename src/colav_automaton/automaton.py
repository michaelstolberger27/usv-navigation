from typing import List, Tuple, Optional

from hybrid_automaton import Automaton, State, Transition
from colav_automaton.guards import G11_and_G12_guard, L1_bar_or_L2_bar_guard, not_G11_guard
from colav_automaton.resets import reset_enter_avoidance, reset_reach_V1, reset_exit_avoidance
from colav_automaton.invariants import is_goal_waypoint_invariant
from colav_automaton.dynamics import (
    waypoint_navigation_dynamics,
    constant_control_dynamics,
)
from colav_automaton.integration import heading_normalizing_integrator_with_self
from colav_controllers import PrescribedTimeController


def ColavAutomaton(
    waypoint_x: float = 10.0,
    waypoint_y: float = 9.0,
    obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
    Cs: float = 2.0,
    a: float = 1.67,
    v: float = 12.0,
    eta: float = 3.5,
    tp: float = 1.0,
    v1_buffer: float = 0.0
) -> Automaton:

    delta = max(5.0, v * tp * 0.5)
    dsafe = Cs + (v * 2) * tp

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
        guards=[G11_and_G12_guard],
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
        guards=[not_G11_guard],
        reset=reset_exit_avoidance,
        priority=1
    ))

    ha = Automaton(
        name="COLAV Automaton",
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
            'pt_controller': PrescribedTimeController(a, v, eta, tp)
        },
        integration_function=heading_normalizing_integrator_with_self
    )

    return ha
