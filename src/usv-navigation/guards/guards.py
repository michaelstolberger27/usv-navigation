# !/usr/bin/python3 
# <---utf-8--->

import numpy as np
from typing import Dict
from hybrid_automaton import Automaton
from colav_automaton.guards.conditions import (
    check_G11_dynamic, check_G12_dynamic, L1_check, L2_check
)

# ============================================================================
# High-Level Guard Functions
# ============================================================================

def G11_and_G12_guard(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
) -> bool:
    """
    Guard for S1 -> S2 transition: Enter collision avoidance
    
    Activates when obstacle is in path (G11) AND close enough (G12).
    Uses the unsafe-set API for dynamic obstacle computation.
        
    Args:
        x: Continuous state [x, y, psi]
        aux_x: Auxiliary states
        u: Control inputs 
        cfg: Configuration containing waypoint, obstacle(s), v, tp, Cs, dsafe
        clk: Clock
        
    Returns:
        bool: True if should enter collision avoidance
    """
    state = x.get_continous_state()
    
    # Convert single obstacle to list format for API compatibility
    if 'obstacles' in cfg and cfg['obstacles']:
        obstacles_list = cfg['obstacles']
    else:
        # Convert legacy single obstacle format to list
        obstacles_list = [(cfg['obstacle_x'], cfg['obstacle_y'], 0.0, 0.0)]
    
    G11 = check_G11_dynamic(
        state[0], state[1],
        cfg['waypoint_x'], cfg['waypoint_y'],
        cfg['v'], cfg['tp'],
        obstacles_list,
        cfg['Cs']
    )
    
    G12 = check_G12_dynamic(
        state[0], state[1], obstacles_list, cfg['dsafe'],
        ship_v=cfg['v'], tp=cfg['tp'], Cs=cfg['Cs']
    )
    
    return G11 and G12


def L1_bar_or_L2_bar_guard(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
) -> bool:
    """
    Guard for S2 -> S3 transition: Enter constant control
    
    Activates when V1 reached (L1) OR V1 is behind (L2)
    
    Docs:
        Corresponds to guard L1 ∨ L2 in ship navigation automaton
        
    Args:
        x: Continuous state [x, y, psi]
        aux_x: Auxiliary states 
        u: Control inputs 
        cfg: Configuration containing delta and ca_controller
        clk: Clock
        
    Returns:
        bool: True if should enter constant control mode 
    """
    state = x.get_continous_state()
    
    # Need virtual waypoint from controller (set by dynamics)
    if 'ca_controller' not in cfg or cfg['ca_controller'] is None:
        return False
    
    if cfg['ca_controller'].virtual_waypoint is None:
        return False
    
    v1_x, v1_y = cfg['ca_controller'].virtual_waypoint
    
    L1 = L1_check(state[0], state[1], v1_x, v1_y, cfg['delta'])
    L2 = L2_check(state[0], state[1], state[2], v1_x, v1_y)
    
    return (not L1) or (not L2)


def not_G11_guard(
    x: Automaton.Runtime.ContinousState,
    aux_x: Dict[str, Automaton.Runtime.AuxiliaryState],
    u: Dict[str, Automaton.Runtime.ControlInput],
    cfg: Dict,
    clk: Automaton.Runtime.Clock
) -> bool:
    """
    Guard for S3 -> S1 transition: Resume waypoint reaching
    
    Activates when LOS to waypoint is clear (G11).
    Uses the unsafe-set API for dynamic obstacle computation.
        
    Args:
        x: Continuous state [x, y, psi]
        aux_x: Auxiliary states 
        u: Control inputs 
        cfg: Configuration containing waypoint, obstacle(s), v, tp, Cs
        clk: Clock
        
    Returns:
        bool: True if LOS is clear and can resume waypoint reaching
    """
    state = x.get_continous_state()
    
    # Convert single obstacle to list format for API compatibility
    if 'obstacles' in cfg and cfg['obstacles']:
        obstacles_list = cfg['obstacles']
    else:
        # Convert legacy single obstacle format to list
        obstacles_list = [(cfg['obstacle_x'], cfg['obstacle_y'], 0.0, 0.0)]
    
    G11 = check_G11_dynamic(
        state[0], state[1],
        cfg['waypoint_x'], cfg['waypoint_y'],
        cfg['v'], cfg['tp'],
        obstacles_list,
        cfg['Cs']
    )
    
    return not G11
