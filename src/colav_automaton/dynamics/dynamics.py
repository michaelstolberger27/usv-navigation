import numpy as np
from hybrid_automaton.automaton_annotations import continuous_dynamics
from hybrid_automaton.automaton_runtime_context import Context
from colav_controllers import PrescribedTimeController, CollisionAvoidanceController
from colav_automaton.unsafe_sets import get_unsafe_set_vertices


def _create_ca_controller(a, v, eta, tp, Cs, dsafe, v1_buffer=0.0):
    """Create CollisionAvoidanceController with unsafe-set vertex provider."""
    def vertex_provider(pos_x, pos_y, obstacles_list, Cs_arg, psi):
        return get_unsafe_set_vertices(
            pos_x, pos_y, obstacles_list, Cs_arg,
            dsf=dsafe, ship_psi=psi, ship_v=v
        )

    return CollisionAvoidanceController(
        a=a, v=v, eta=eta, tp=tp, Cs=Cs,
        vertex_provider=vertex_provider,
        v1_buffer=v1_buffer
    )


@continuous_dynamics
def S1_waypoint_reaching_dynamics(ctx: Context) -> np.ndarray:
    """
    S1: Waypoint reaching - navigate to waypoint using prescribed-time control.

    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)

    Uses prescribed-time controller to guarantee heading convergence to LOS.
    Navigates to the top of the waypoints stack (LIFO).
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    t = ctx.clk.get_time_elapsed_since_transition()

    if 'pt_controller' not in cfg:
        cfg['pt_controller'] = PrescribedTimeController(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp']
        )

    target_x, target_y = cfg['waypoints'][-1]

    return cfg['pt_controller'].compute_dynamics(
        t, state[0], state[1], state[2],
        target_x, target_y
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
    Pushes V1 onto the waypoints stack when computed.
    """
    state = ctx.x.latest()
    cfg = ctx.cfg
    t = ctx.clk.get_time_elapsed_since_transition()

    if 'ca_controller' not in cfg:
        cfg['ca_controller'] = _create_ca_controller(
            cfg['a'], cfg['v'], cfg['eta'], cfg['tp'], cfg['Cs'],
            cfg['dsafe'],
            v1_buffer=cfg.get('v1_buffer', 0.0)
        )

    # Set virtual waypoint V1 once when entering S2 (not every timestep)
    if cfg['ca_controller'].virtual_waypoint is None:
        cfg['ca_controller'].set_virtual_waypoint(
            state[0], state[1], state[2],
            cfg['obstacles']
        )

        # Push V1 onto waypoints stack
        if cfg['ca_controller'].virtual_waypoint is not None:
            cfg['waypoints'].append(cfg['ca_controller'].virtual_waypoint)

    return cfg['ca_controller'].compute_dynamics(t, state[0], state[1], state[2])


@continuous_dynamics
def S3_constant_control_dynamics(ctx: Context) -> np.ndarray:
    """
    S3: Constant heading - maintain straight-line motion after avoidance.

    State x: [x, y, psi]
        - x, y: position (m)
        - psi: heading (rad)

    Holds the current heading with zero turning rate until LOS is clear.
    """
    state = ctx.x.latest()
    cfg = ctx.cfg

    return cfg['ca_controller'].compute_constant_dynamics(
        state[0], state[1], state[2]
    )
