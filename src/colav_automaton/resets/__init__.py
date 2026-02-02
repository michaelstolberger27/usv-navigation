""" 
Resets for hybrid-automaton state transitions.
"""
from .resets import (
    reset_enter_avoidance,
    reset_exit_avoidance
)

__author__ = "Ryan McKee <r.mckee@liverpool.ac.uk>"
__verion__ = "0.0.1"
__description__ = "resets for hybrid-automaton transitions that will takes all inputs and " \
                "output update continous states (agent continous state scalers, auxielary continous" \
                "states dictionary) these resets are specifically for colav-automaton"

__all__ = [
    "reset_enter_avoidance",
    "reset_exit_avoidance"
]