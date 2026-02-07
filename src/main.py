import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from colav_automaton import ColavAutomaton
from colav_automaton.integration import normalize_heading_in_results # For visualisation
from hybrid_automaton import Automaton
from hybrid_automaton_runner import AutomatonRunner
import asyncio
import numpy as np
from hybrid_automaton_evaluation.figure_generator import continuous_states_over_time_fig, auxiliary_states_over_time_fig, automaton_states_over_time, transitions_times_over_time_fig

async def main():
    # Ship Navigation Automaton with prescribed-time control
    ha: Automaton = ColavAutomaton(
        waypoint_x=10.0,      # Target waypoint
        waypoint_y=9.0,
        obstacles=[(5.0, 4.5, 0.0, 0.0)],  # (x, y, speed, heading)
        Cs=2.0,               # Safe distance from obstacle
        a=1.67,               # System parameter
        v=12.0,               # Ship velocity (m/s) - constant
        eta=3.5,              # Controller gain
        tp=1.0                # Prescribed time
    )
    
    # Initial state: [x, y, psi] -
    x0 = np.array([0.0, 0.0, 0.0], dtype=float)  # Start at origin, heading 0 rad
    
    # set to None or empty dict
    aux_t0 = {}

    print(str(ha))
    print(repr(ha))
    
    ha_runner: AutomatonRunner = AutomatonRunner(ha, sampling_rate=0.001)
    await ha_runner.run(
        x0=x0,
        aux_x0=aux_t0,
        duration=15.0,
        real_time_mode=False,
        integrate=True,
        dt=0.1,
        collect_control=False
    )
    
    results = ha_runner.get_results()

    # Normalize heading angles to [-π, π] for visualization
    results = normalize_heading_in_results(results)

    print(results)

    # Generate plots - pass the specific lists, not the full results dict
    continuous_states_over_time_fig(results['continuous_states'])
    automaton_states_over_time(results['automaton_states'])
    transitions_times_over_time_fig(results['transition_times'])

# Run the event loop
asyncio.run(main())
