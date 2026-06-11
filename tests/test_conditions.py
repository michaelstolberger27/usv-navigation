"""
Unit tests for the guard condition functions (paper eq 13-27).

Geometry convention used throughout: the ego vessel sits at the origin
heading along +x (psi = 0), so starboard is -y and port is +y.
Obstacles are (ox, oy, ov, o_psi) tuples.
"""
import numpy as np
import pytest

from colav_automaton.guards.conditions import (
    _F,
    classify_encounter,
    compute_risk_index,
    G11_check,
    G12_check,
    G22_check,
    G23_check,
    L1_check,
    L2_check,
)

CS = 300.0
V = 5.0
TP = 3.0
DSAFE = CS + V * TP


# ---------------------------------------------------------------- _F (eq 20)

class TestRiskFunctionF:
    def test_max_risk_at_or_below_beta1(self):
        assert _F(100.0, 148.0, 463.0) == 1.0
        assert _F(148.0, 148.0, 463.0) == 1.0

    def test_zero_risk_above_beta2(self):
        assert _F(464.0, 148.0, 463.0) == 0.0
        assert _F(1e9, 148.0, 463.0) == 0.0

    def test_half_risk_at_midpoint(self):
        beta1, beta2 = 148.0, 463.0
        mid = (beta1 + beta2) / 2.0
        assert _F(mid, beta1, beta2) == pytest.approx(0.5)

    def test_monotonically_decreasing(self):
        beta1, beta2 = 120.0, 240.0
        zs = np.linspace(beta1, beta2, 50)
        fs = [_F(z, beta1, beta2) for z in zs]
        assert all(a >= b for a, b in zip(fs, fs[1:]))

    def test_range_is_unit_interval(self):
        for z in np.linspace(0.0, 600.0, 61):
            assert 0.0 <= _F(z, 148.0, 463.0) <= 1.0


# ------------------------------------------------- risk index / G22 (eq 19-21)

class TestRiskIndex:
    def test_no_obstacles_is_zero(self):
        assert compute_risk_index(0, 0, 0.0, [], V, CS) == 0.0

    def test_receding_obstacle_is_zero(self):
        # Obstacle abeam, sailing directly away: negative TCPA is skipped.
        obstacles = [(1000.0, -800.0, 8.0, -np.pi / 2)]
        assert compute_risk_index(0, 0, 0.0, obstacles, V, CS) == 0.0

    def test_close_head_on_obstacle_is_high_risk(self):
        # Head-on at 400 m: DCPA ~ 0, TCPA = 400/(5+5) = 40 s.
        obstacles = [(400.0, 0.0, 5.0, np.pi)]
        ri = compute_risk_index(0, 0, 0.0, obstacles, V, CS)
        assert ri >= 0.35

    def test_distant_crossing_obstacle_is_low_risk(self):
        # 10 km away: every F term is zero.
        obstacles = [(10000.0, -3000.0, 5.0, np.pi / 2)]
        ri = compute_risk_index(0, 0, 0.0, obstacles, V, CS)
        assert ri == 0.0

    def test_result_in_unit_interval(self):
        geometries = [
            [(400.0, 0.0, 5.0, np.pi)],
            [(1000.0, -800.0, 8.0, np.pi / 2)],
            [(50.0, 50.0, 0.0, 0.0)],
            [(2000.0, 100.0, 3.0, np.pi)],
        ]
        for obstacles in geometries:
            ri = compute_risk_index(0, 0, 0.0, obstacles, V, CS)
            assert 0.0 <= ri <= 1.0

    def test_g22_thresholds_against_risk_index(self):
        obstacles = [(400.0, 0.0, 5.0, np.pi)]
        ri = compute_risk_index(0, 0, 0.0, obstacles, V, CS)
        assert G22_check(0, 0, 0.0, obstacles, V, CS, K=ri - 0.01)
        assert not G22_check(0, 0, 0.0, obstacles, V, CS, K=ri + 0.01)


# --------------------------------------------------------------- L1 (eq 15)

class TestL1:
    def test_true_when_beyond_delta(self):
        assert L1_check(0, 0, 100.0, 0.0, delta=10.0)

    def test_false_when_within_delta(self):
        assert not L1_check(0, 0, 5.0, 0.0, delta=10.0)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            L1_check(0, 0, 100.0, 0.0, delta=0.0)


# --------------------------------------------------------------- L2 (eq 16)

class TestL2:
    def test_v1_ahead(self):
        assert L2_check(0, 0, 0.0, 100.0, 0.0)

    def test_v1_behind(self):
        assert not L2_check(0, 0, 0.0, -100.0, 0.0)

    def test_v1_exactly_abeam_is_not_ahead(self):
        # eq 16 uses a strict inequality at +/- pi/2.
        assert not L2_check(0, 0, 0.0, 0.0, 100.0)

    def test_respects_heading(self):
        # Heading +y: a point on +y is ahead, a point on +x is abeam.
        assert L2_check(0, 0, np.pi / 2, 0.0, 100.0)
        assert not L2_check(0, 0, np.pi / 2, 100.0, 0.0)


# ------------------------------------------------------- encounter classifier

class TestClassifyEncounter:
    def test_no_obstacles(self):
        assert classify_encounter(0.0, [], 0, 0) == "none"

    def test_head_on(self):
        obstacles = [(1000.0, 0.0, 5.0, np.pi)]
        assert classify_encounter(0.0, obstacles, 0, 0) == "head_on"

    def test_crossing_from_port(self):
        obstacles = [(300.0, 400.0, 5.0, -np.pi / 2)]
        assert classify_encounter(0.0, obstacles, 0, 0) == "crossing_from_port"

    def test_crossing_from_starboard(self):
        obstacles = [(300.0, -400.0, 5.0, np.pi / 2)]
        assert classify_encounter(0.0, obstacles, 0, 0) == "crossing_from_starboard"

    def test_overtaking(self):
        obstacles = [(-500.0, 0.0, 8.0, 0.0)]
        assert classify_encounter(0.0, obstacles, 0, 0) == "overtaking"

    def test_uses_closest_obstacle(self):
        obstacles = [
            (300.0, 400.0, 5.0, -np.pi / 2),   # port crossing, 500 m
            (5000.0, 0.0, 5.0, np.pi),          # head-on, 5 km
        ]
        assert classify_encounter(0.0, obstacles, 0, 0) == "crossing_from_port"


# -------------------------------------------------------------- G11 (eq 13)

class TestG11:
    def test_no_obstacles(self):
        assert G11_check(0, 0, 0.0, 2000.0, 0.0, V, TP, [], CS) is False

    def test_invalid_parameters_raise(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0)]
        with pytest.raises(ValueError):
            G11_check(0, 0, 0.0, 2000.0, 0.0, 0.0, TP, obstacles, CS)

    def test_static_obstacle_on_los_blocks(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0)]
        assert G11_check(0, 0, 0.0, 2000.0, 0.0, V, TP, obstacles, CS) is True

    def test_static_obstacle_far_abeam_is_clear(self):
        obstacles = [(500.0, 5000.0, 0.0, 0.0)]
        assert G11_check(0, 0, 0.0, 2000.0, 0.0, V, TP, obstacles, CS) is False


# -------------------------------------------------------------- G12 (eq 14)

class TestG12:
    def test_obstacle_within_dsafe(self):
        obstacles = [(DSAFE - 50.0, 0.0, 0.0, 0.0)]
        assert G12_check(0, 0, 0.0, obstacles, V, CS, DSAFE, TP) is True

    def test_stationary_obstacle_beyond_dsafe(self):
        obstacles = [(DSAFE + 50.0, 0.0, 0.0, 0.0)]
        assert G12_check(0, 0, 0.0, obstacles, V, CS, DSAFE, TP) is False

    def test_closing_obstacle_extends_effective_dsafe(self):
        # Same range as above but closing fast: effective dsafe grows by
        # closing_speed * tp = 60 m, so the 50 m margin is inside it.
        obstacles = [(DSAFE + 50.0, 0.0, 20.0, np.pi)]
        assert G12_check(0, 0, 0.0, obstacles, V, CS, DSAFE, TP) is True


# -------------------------------------------------------------- G23 (eq 27)

class TestG23:
    def test_no_obstacles(self):
        assert G23_check(0, 0, 0.0, 2000.0, 0.0, V, TP, [], CS) is False

    def test_degenerate_parameters_do_not_block(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0)]
        assert G23_check(0, 0, 0.0, 2000.0, 0.0, 0.0, TP, obstacles, CS) is False

    def test_obstacle_on_los_blocks_resume(self):
        obstacles = [(800.0, 0.0, 0.0, 0.0)]
        assert G23_check(0, 0, 0.0, 2000.0, 0.0, V, TP, obstacles, CS) is True

    def test_static_only_ignores_converging_off_path_obstacle(self):
        # The resume check is deliberately static-only: an obstacle whose
        # *current* Cs circle is clear of the LOS does not block resume,
        # even though it is converging.  This blind spot is what the
        # K_off risk hysteresis in not_G23_guard compensates for.
        obstacles = [(1000.0, -800.0, 8.0, np.pi / 2)]
        assert G23_check(0, 0, 0.0, 2000.0, 0.0, V, TP, obstacles, CS) is False
