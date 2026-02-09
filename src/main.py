import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from colav_automaton import ColavAutomaton
from colav_automaton.controllers import HeadingControlProvider
from hybrid_automaton import Automaton, RunResult
import asyncio
import numpy as np

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

    print(str(ha))
    print(repr(ha))

    # Controller runs asynchronously to the automaton
    controller = HeadingControlProvider(ha)

    results: RunResult = await ha.activate(
        initial_continuous_state=x0,
        initial_control_input_states={'u': np.array([0.0])},
        enable_real_time_mode=False,
        enable_self_integration=True,
        delta_time=0.1,
        timeout_sec=15.0,
        continuous_state_sampler_enabled=True,
        continuous_state_sampler_rate=100,
        control_states_provider=controller,
        control_states_provision_rate=100,
    )

    print(results)

# Run the event loop
asyncio.run(main())
