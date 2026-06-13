"""
Tests for the evaluation metrics helpers.

Focus on the two corrected behaviours: compute_cpa must keep scanning
past trajectory gaps (it used to break on the first missing state), and
check_goal_reached on the oriented goal rectangle.

classify_encounter is imported from the controller's own implementation
and is covered by test_conditions, so it is not re-tested here.
"""
import numpy as np

from commonocean_integration.evaluation.metrics import (
    check_goal_reached,
    compute_cpa,
)


class _FakeState:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)


class _FakeObstacle:
    """Returns a state per step index, or None for gap steps."""

    def __init__(self, states_by_step):
        self._states = states_by_step

    def state_at_time(self, t):
        return self._states.get(t)


class TestComputeCPA:
    def test_basic_minimum(self):
        ego = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
        obs = _FakeObstacle({0: _FakeState([0.0, 100.0]),
                             1: _FakeState([10.0, 30.0]),   # closest
                             2: _FakeState([20.0, 80.0])})
        d, step = compute_cpa(ego, obs, max_steps=3)
        assert d == 30.0
        assert step == 1

    def test_gap_does_not_truncate(self):
        # Step 1 has no obstacle state. The old code broke here and would
        # miss the true CPA at step 2; the fix keeps scanning.
        ego = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
        obs = _FakeObstacle({0: _FakeState([0.0, 100.0]),
                             1: None,
                             2: _FakeState([20.0, 5.0])})   # closest, after gap
        d, step = compute_cpa(ego, obs, max_steps=3)
        assert d == 5.0
        assert step == 2

    def test_no_states_returns_inf(self):
        ego = [[0.0, 0.0], [10.0, 0.0]]
        obs = _FakeObstacle({})
        d, step = compute_cpa(ego, obs, max_steps=2)
        assert d == float("inf")


class TestCheckGoalReached:
    def test_inside_axis_aligned_rect(self):
        traj = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]
        reached, step = check_goal_reached(
            traj, np.array([100.0, 0.0]), 20.0, 20.0, 0.0)
        assert reached
        assert step == 2

    def test_never_enters_rect(self):
        traj = [[0.0, 0.0], [50.0, 50.0]]
        reached, step = check_goal_reached(
            traj, np.array([100.0, 0.0]), 10.0, 10.0, 0.0)
        assert not reached
        assert step is None

    def test_respects_rectangle_orientation(self):
        # A long thin goal rotated 90 deg: a point offset along the
        # rotated long axis is inside; the same offset along the short
        # axis is outside.
        center = np.array([0.0, 0.0])
        length, width, orient = 40.0, 4.0, np.pi / 2
        inside = [[0.0, 15.0]]   # 15 m along rotated long axis
        outside = [[15.0, 0.0]]  # 15 m along rotated short axis
        assert check_goal_reached(inside, center, length, width, orient)[0]
        assert not check_goal_reached(outside, center, length, width, orient)[0]
