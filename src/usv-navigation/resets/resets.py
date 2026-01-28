from hybrid_automaton import Automaton
from typing import Dict

def reset_enter_avoidance(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
):
    """
    Reset when entering S2 (collision avoidance mode).

    Clears any previous collision avoidance state to ensure fresh
    computation of virtual waypoint V1.

    Args:
        x: Continuous state 
        aux_x: Auxiliary states
        u: Control inputs 
        cfg: Configuration containing ca_controller
        clk: Clock 

    Returns:
        Tuple[x, aux_x, u]: Unchanged states
    """
    if 'ca_controller' in cfg and cfg['ca_controller'] is not None:
        cfg['ca_controller'].reset()
    return x, aux_x, u


def reset_exit_avoidance(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
):
    """
    Reset when exiting S3 back to S1 (resume waypoint reaching).

    Clears collision avoidance state to prepare for normal waypoint navigation.

    Args:
        x: Continuous state 
        aux_x: Auxiliary states 
        u: Control inputs 
        cfg: Configuration containing ca_controller
        clk: Clock

    Returns:
        Tuple[x, aux_x, u]: Unchanged states
    """
    if 'ca_controller' in cfg and cfg['ca_controller'] is not None:
        cfg['ca_controller'].reset()
    return x, aux_x, u
