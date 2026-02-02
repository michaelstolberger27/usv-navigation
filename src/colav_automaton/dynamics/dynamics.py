import numpy as np
from hybrid_automaton.automaton_annotations import continuous_dynamics
from hybrid_automaton.automaton_runtime_context import Context
from colav_controllers import PrescribedTimeController
from colav_controllers import CollisionAvoidanceController as _BaseCollisionAvoidanceController
from colav_automaton.unsafe_sets import get_unsafe_set_vertices


def _create_ca_controller(a, v, eta, tp, Cs, dsafe, ship_v, v1_buffer=0.0):
    """Create CollisionAvoidanceController with unsafe-set vertex provider."""
    # Create wrapper that passes dsafe and ship heading to get_unsafe_set_vertices
    def vertex_provider_with_dynamics(pos_x, pos_y, obstacles_list, Cs_arg, psi):
        return get_unsafe_set_vertices(
            pos_x, pos_y, obstacles_list, Cs_arg,
            dsf=dsafe, ship_psi=psi, ship_v=ship_v
        )

    return _BaseCollisionAvoidanceController(
        a=a, v=v, eta=eta, tp=tp, Cs=Cs,
        vertex_provider=vertex_provider_with_dynamics,
        v1_buffer=v1_buffer
    )


# Dynamics from paper
@continuous_dynamics
def S1_waypoint_reaching_dynamics(ctx: Context) -> np.ndarray:
    """
    S1: Waypoint reaching - navigate to waypoint using prescribed-time control.

    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)

    Uses prescribed-time controller to guarantee heading convergence to LOS.
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    t = ctx.clk.get_time_elapsed_since_transition()

    # Get or create controller (stored in cfg to persist across calls)
    if 'pt_controller' not in cfg:
        cfg['pt_controller'] = PrescribedTimeController(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp']
        )

    return cfg['pt_controller'].compute_dynamics(
        t, state[0], state[1], state[2],
        cfg['waypoint_x'], cfg['waypoint_y']
    )


@continuous_dynamics
def S2_collision_avoidance_dynamics(ctx: Context) -> np.ndarray:
    """
    S2: Collision avoidance - navigate to V1 using prescribed-time control.

    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)

    Computes virtual waypoint V1 (starboard vertex of unsafe set) and
    applies prescribed-time control to reach it.

    Supports both single obstacle (legacy) and multiple obstacles (dynamic).
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    t = ctx.clk.get_time_elapsed_since_transition()

    # Get or create controller
    if 'ca_controller' not in cfg:
        cfg['ca_controller'] = _create_ca_controller(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp'], cfg['Cs'],
            cfg['dsafe'], cfg['v'],
            v1_buffer=cfg.get('v1_buffer', 0.0)
        )

    # Set virtual waypoint V1 once when entering S2 (not every timestep)
    if cfg['ca_controller'].virtual_waypoint is None:
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


@continuous_dynamics
def S3_constant_control_dynamics(ctx: Context) -> np.ndarray:
    """
    S3: Constant control - hold last control value from S2.

    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)

    Maintains the control input that was active when transitioning from S2.
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    return cfg['ca_controller'].compute_constant_dynamics(
        state[0], state[1], state[2]
    )