from typing import Dict
from hybrid_automaton import Automaton
from colav_controllers import PrescribedTimeController
from colav_controllers import CollisionAvoidanceController as _BaseCollisionAvoidanceController
from colav_automaton.unsafe_sets import get_unsafe_set_vertices


def _create_ca_controller(a, v, eta, tp, Cs, v1_buffer=0.0):
    """Create CollisionAvoidanceController with unsafe-set vertex provider."""
    return _BaseCollisionAvoidanceController(
        a=a, v=v, eta=eta, tp=tp, Cs=Cs,
        vertex_provider=get_unsafe_set_vertices,
        v1_buffer=v1_buffer
    )


# Dynamics from paper
def S1_waypoint_reaching_dynamics(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
):
    """
    S1: Waypoint reaching - navigate to waypoint using prescribed-time control.
    
    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)
    
    Uses prescribed-time controller to guarantee heading convergence to LOS.
    """
    state = x.get_continous_state()
    t = clk.get_time_elapsed_since_last_transition()

    # Get or create controller (stored in cfg to persist across calls)
    if 'pt_controller' not in cfg:
        cfg['pt_controller'] = PrescribedTimeController(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp']
        )

    return cfg['pt_controller'].compute_dynamics(
        t, state[0], state[1], state[2],
        cfg['waypoint_x'], cfg['waypoint_y']
    )


def S2_collision_avoidance_dynamics(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
):
    """
    S2: Collision avoidance - navigate to V1 using prescribed-time control.
    
    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)
    
    Computes virtual waypoint V1 (starboard vertex of unsafe set) and
    applies prescribed-time control to reach it.
    
    Supports both single obstacle (legacy) and multiple obstacles (dynamic).
    """
    state = x.get_continous_state()
    t = clk.get_time_elapsed_since_last_transition()

    # Get or create controller
    if 'ca_controller' not in cfg:
        cfg['ca_controller'] = _create_ca_controller(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp'], cfg['Cs'],
            v1_buffer=cfg.get('v1_buffer', 0.0)
        )

    # Set virtual waypoint V1 - choose mode based on obstacles configuration
    if 'obstacles' in cfg and cfg['obstacles']:
        # Dynamic mode: multiple obstacles
        cfg['ca_controller'].set_virtual_waypoint_dynamic(
            state[0], state[1], state[2],
            cfg['obstacles']
        )
    else:
        # Legacy mode: single obstacle
        cfg['ca_controller'].set_virtual_waypoint(
            state[0], state[1], state[2],
            cfg['obstacle_x'], cfg['obstacle_y']
        )

    return cfg['ca_controller'].compute_dynamics(t, state[0], state[1], state[2])


def S3_constant_control_dynamics(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
):
    """
    S3: Constant control - hold last control value from S2.
    
    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)
    
    Maintains the control input that was active when transitioning from S2.
    """
    state = x.get_continous_state()

    return cfg['ca_controller'].compute_constant_dynamics(
        state[0], state[1], state[2]
    )
