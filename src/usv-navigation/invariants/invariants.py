from typing import List, Dict
import numpy as np
from hybrid_automaton import Automaton

def is_goal_waypoint_invariant(x: np.array, aux_x: Dict[str, Automaton.Runtime.AuxiliaryState], u: Dict[str, Automaton.Runtime.ControlInput], cfg: Dict, clk: Automaton.Runtime.Clock) -> bool:
    if cfg["waypoints"] == 1: 
        return True
    return False
    