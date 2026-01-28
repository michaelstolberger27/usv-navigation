from finite_time_control import HeadingFTC
from typing import Tuple, Callable
from typing import Dict

from hybrid_automaton import Automaton
import numpy as np

def integrate_simple_vessel_kinematics(x: np.array, aux_x: Dict[str, Automaton.Runtime.AuxiliaryState], u: Dict[str, Automaton.Runtime.ControlInput], cfg: Dict, clk: Automaton.Runtime.Clock) -> Tuple[float, float, float]:
    """
    used for simulating the integration of continuous dynamics
    into a finite time controller

    Args: 
        x: float | vector
            represents agetn continous state we are 
            executing commands on for the hybrid automaton,
            for colav its a vector: [x,y,θ,v,θ_rate]
        aux_x: Dict
            a dictionary containing auxielary continous states
            keys and their values, these are continous states
            which retrieves from external sources like sensors
            and so on, for colav we have {'waypoints': [(10.0, 10.0), (100.0, 100.0)]}
        ctx: Dict
            contextual information regarding the state of the automaton
            this will contain current discrete state (Mode | Q), 
            time since last transition time active and configuration 
            information regarding the current implementation of the automaton
        u: Dict
            additional command influeces, this can be stuff like 
            how much sway their is for a boat we need to coutneract.
        dt: float
            delta of time between last integration, which is used to calculate
            the rate delta of the integration we are going to be calculating
            for. 

    Outputs: 
        list (a vector of scalars in np.array format representing the x,y,theta state of automaton) 
    """
    ctrl = HeadingFTC()
    ctrl.update_x_state(x)
    desired_waypoint = aux_x['waypoints'][0] # should get the first waypoint in the list
    ctrl.update_desired_state(desired_waypoint)
    desired_yaw_rate = ctrl.update(dt=dt, ctrl_saturation=True)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def heading_normalizing_integrator(state: np.ndarray, xdot: np.ndarray, dt: float) -> np.ndarray:
    """
    Integration function compatible with hybrid_automaton's Automaton.

    This function matches the signature expected by Automaton.Runtime.ContinousState:
        integration_function(state, xdot, dt) -> new_state

    Performs Euler integration and normalizes the heading angle (index 2) to [-π, π].

    Args:
        state: Current state vector [x, y, psi, ...]
        xdot: State derivatives [dx/dt, dy/dt, dpsi/dt, ...]
        dt: Integration timestep

    Returns:
        New state with normalized heading
    """
    # Debug print to see what's being passed
    # print(f"  [integrator] state={state}, xdot={xdot}, dt={dt}")

    x_new = state + xdot * dt

    # Normalize heading at index 2 (psi)
    if len(x_new) > 2:
        x_new[2] = normalize_angle(x_new[2])

    return x_new


def heading_normalizing_integrator_with_self(*args):
    """
    Wrapper that handles both 3-arg and 4-arg calls.
    Some versions of hybrid_automaton may pass self as first argument.
    """
    if len(args) == 3:
        state, xdot, dt = args
    elif len(args) == 4:
        # If 4 args, first might be self or state
        _, state, xdot, dt = args
    else:
        raise ValueError(f"Expected 3 or 4 arguments, got {len(args)}")

    x_new = state + xdot * dt
    if len(x_new) > 2:
        x_new[2] = normalize_angle(x_new[2])
    return x_new


def integrate_ship_dynamics_with_heading_normalization(
    x_current: np.ndarray,
    dynamics_fn: Callable,
    dt: float,
    aux_x: Dict,
    u: Dict,
    cfg: Dict,
    clk
) -> np.ndarray:
    """
    Custom integration method for ship dynamics that normalises heading angle.

    This integrator is designed for state vectors where index 2 is the heading (psi).
    After integration, it wraps the heading to [-π, π] to prevent unbounded growth.

    Args:
        x_current: Current state [x, y, psi, ...]
        dynamics_fn: Function that returns state derivatives
        dt: Integration timestep
        aux_x: Auxiliary states
        u: Control inputs
        cfg: Configuration dictionary
        clk: Clock object

    Returns:
        Updated state with normalized heading
    """
    # Create a wrapper for the current state
    class StateWrapper:
        def __init__(self, state):
            self._state = state
        def get_continous_state(self):
            return self._state

    x_wrapped = StateWrapper(x_current)

    # Compute derivatives using the dynamics function
    dx_dt = dynamics_fn(x_wrapped, aux_x, u, cfg, clk)

    # Euler integration
    x_next = x_current + dx_dt * dt

    # Normalize heading (assuming it's at index 2)
    # State structure: [x, y, psi, ...]
    if len(x_next) > 2:
        x_next[2] = normalize_angle(x_next[2])

    return x_next


def normalize_heading_in_results(results: Dict) -> Dict:
    """
    Post-process simulation results to normalise all heading values to [-π, π].

    This function modifies the continuous_states in the results dictionary
    to ensure all heading angles (at index 2) are wrapped to the range [-π, π].

    Args:
        results: Dictionary containing 'continuous_states' key with list of (time, state) tuples

    Returns:
        Modified results dictionary with normalised headings

    Note:
        This is a post-processing step that fixes visualization and analysis.
        It does NOT affect the actual simulation - headings may still grow
        unbounded during integration if the integrator doesn't normalise.
    """
    if 'continuous_states' in results:
        for i, (t, state) in enumerate(results['continuous_states']):
            if len(state) > 2:
                # Normalize heading at index 2
                state[2] = normalize_angle(state[2])

    return results
