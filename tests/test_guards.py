"""
Tests for the guard wrappers, in particular the K_off risk hysteresis on
the S3 -> S1 resume transition (extension beyond the paper, see
not_G23_guard's docstring).

Guards are @guard-decorated into _Annotation objects whose __call__
type-checks for the runtime Context, so tests invoke the undecorated
function via `.func` with a minimal stand-in context.
"""
import numpy as np
import pytest

from colav_automaton.guards import (
    G11_and_G22_guard,
    L1_bar_or_L2_bar_guard,
    not_G23_guard,
)
from colav_automaton.guards.conditions import compute_risk_index, G23_check


class _FakeContinuousState:
    def __init__(self, state):
        self._state = np.asarray(state, dtype=float)

    def latest(self):
        return self._state


class _FakeCtx:
    def __init__(self, state, configuration):
        self.continuous_state = _FakeContinuousState(state)
        self.configuration = configuration


def make_ctx(obstacles, psi=0.0, **overrides):
    cfg = {
        'waypoints': [(2000.0, 0.0)],
        'obstacles': obstacles,
        'v': 5.0,
        'tp': 3.0,
        'Cs': 300.0,
        'dsafe': 315.0,
        'delta': 10.0,
    }
    cfg.update(overrides)
    return _FakeCtx([0.0, 0.0, psi], cfg)


# Off the LOS (800 m abeam) but converging: crosses the ego's track ahead.
CONVERGING_OFF_PATH = [(1000.0, -800.0, 8.0, np.pi / 2)]
# Same position, sailing away.
RECEDING_OFF_PATH = [(1000.0, -800.0, 8.0, -np.pi / 2)]
# Parked on the LOS to the waypoint.
ON_LOS = [(800.0, 0.0, 0.0, 0.0)]


class TestNotG23Hysteresis:
    def test_geometry_premises(self):
        # The hysteresis tests below rely on this geometry: LOS clear
        # (static G23 false) while the predictive risk index stays high.
        assert G23_check(0, 0, 0.0, 2000.0, 0.0, 5.0, 3.0,
                         CONVERGING_OFF_PATH, 300.0) is False
        ri = compute_risk_index(0, 0, 0.0, CONVERGING_OFF_PATH, 5.0, 300.0)
        assert ri >= 0.25

    def test_resume_when_clear_and_low_risk(self):
        ctx = make_ctx(RECEDING_OFF_PATH)
        assert not_G23_guard.func(ctx)

    def test_hysteresis_blocks_resume_while_risk_high(self):
        # LOS is clear, but the converging obstacle keeps RI >= K_off:
        # without the hysteresis this is the T-1022 chattering case.
        ctx = make_ctx(CONVERGING_OFF_PATH)
        assert not not_G23_guard.func(ctx)

    def test_k_off_at_or_above_one_recovers_pure_not_g23(self):
        ctx = make_ctx(CONVERGING_OFF_PATH, K_off=1.01)
        assert not_G23_guard.func(ctx)

    def test_blocked_los_blocks_resume_regardless_of_k_off(self):
        ctx = make_ctx(ON_LOS, K_off=1.01)
        assert not not_G23_guard.func(ctx)

    def test_no_obstacles_resumes(self):
        ctx = make_ctx([])
        assert not_G23_guard.func(ctx)


class TestG11AndG22Guard:
    def test_no_obstacles(self):
        ctx = make_ctx([])
        assert not G11_and_G22_guard.func(ctx)

    def test_head_on_obstacle_triggers(self):
        ctx = make_ctx([(400.0, 0.0, 5.0, np.pi)])
        assert G11_and_G22_guard.func(ctx)

    def test_obstacle_far_abeam_does_not_trigger(self):
        ctx = make_ctx([(500.0, 5000.0, 0.0, 0.0)])
        assert not G11_and_G22_guard.func(ctx)


class TestL1BarOrL2BarGuard:
    def test_no_v1_on_stack(self):
        # Only the goal waypoint: guard cannot fire.
        ctx = make_ctx([(400.0, 0.0, 5.0, np.pi)])
        assert not L1_bar_or_L2_bar_guard.func(ctx)

    def test_v1_far_ahead_keeps_avoiding(self):
        ctx = make_ctx([(400.0, 0.0, 5.0, np.pi)])
        ctx.configuration['waypoints'] = [(2000.0, 0.0), (500.0, -200.0)]
        assert not L1_bar_or_L2_bar_guard.func(ctx)

    def test_v1_reached_fires(self):
        ctx = make_ctx([(400.0, 0.0, 5.0, np.pi)])
        ctx.configuration['waypoints'] = [(2000.0, 0.0), (3.0, 0.0)]
        assert L1_bar_or_L2_bar_guard.func(ctx)

    def test_v1_behind_fires(self):
        ctx = make_ctx([(400.0, 0.0, 5.0, np.pi)])
        ctx.configuration['waypoints'] = [(2000.0, 0.0), (-500.0, 0.0)]
        assert L1_bar_or_L2_bar_guard.func(ctx)
