"""
Guards the CPA-convention normalization (colav_automaton._compat).

Environment-agnostic: these pass whether the installed colav_unsafe_set
build already matches the validated convention (normalization is a
no-op) or had to be rebound. If they fail, every risk-index and
unsafe-set decision in the controller is suspect.
"""

import numpy as np
import pytest

import colav_automaton  # noqa: F401  (import applies the normalization)
from colav_automaton.controllers.unsafe_sets import (
    _create_agent,
    _create_obstacles,
    calculate_obstacle_metrics_for_agent,
)


def _metrics(agent_state, obstacle):
    agent = _create_agent(*agent_state, Cs=1.0)
    return calculate_obstacle_metrics_for_agent(
        agent, _create_obstacles([obstacle], Cs=1.0))[0]


class TestDcpaConvention:
    def test_collision_course_has_zero_dcpa(self):
        # Head-on at 10 m closing at 2 m/s: impact in 5 s, DCPA = 0.
        m = _metrics((0.0, 0.0, 0.0, 1.0), (10.0, 0.0, 1.0, np.pi))
        assert m.tcpa == pytest.approx(5.0)
        assert m.dcpa == pytest.approx(0.0, abs=1e-9)

    def test_offset_crossing_dcpa_equals_lateral_miss(self):
        # Obstacle passes 3 m abeam of a stationary agent: DCPA = 3.
        m = _metrics((0.0, 0.0, 0.0, 0.0), (10.0, 3.0, 1.0, np.pi))
        assert m.dcpa == pytest.approx(3.0)
        assert m.tcpa == pytest.approx(10.0)
