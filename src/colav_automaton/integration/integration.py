from typing import Callable, Dict

import numpy as np


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
