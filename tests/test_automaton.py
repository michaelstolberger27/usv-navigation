"""
Tests for automaton construction and the delta computation (paper eq 9).
"""
import pytest

from colav_automaton import ColavAutomaton
from colav_automaton.automaton import _compute_delta


class TestComputeDelta:
    def test_non_binding_constraint_uses_heuristic(self):
        # Default-style parameters: m - pi - eta*pi/(a*tp^eta) < 0, so the
        # input constraint is non-binding and the heuristic applies.
        v, a, eta, tp, m = 12.0, 1.67, 3.5, 1.0, 3.0
        assert _compute_delta(v, a, eta, tp, m) == max(5.0, v * tp * 0.5)

    def test_heuristic_floor_at_low_speed(self):
        assert _compute_delta(0.5, 1.67, 3.5, 1.0, 3.0) == 5.0

    def test_binding_constraint_formula(self):
        # Large m makes the denominator positive: delta = 2v / (a * inner).
        v, a, eta, tp, m = 12.0, 1.67, 3.5, 2.0, 20.0
        delta = _compute_delta(v, a, eta, tp, m)
        assert delta >= 1.0
        import numpy as np
        inner = m - np.pi - eta * np.pi / (a * tp ** eta)
        assert delta == pytest.approx(max(2.0 * v / (a * inner), 1.0))

    def test_always_positive(self):
        for v in (0.1, 5.0, 12.0, 30.0):
            for tp in (0.5, 1.0, 3.0):
                assert _compute_delta(v, 1.67, 3.5, tp, 3.0) > 0.0


class TestColavAutomatonConstruction:
    def test_default_construction(self):
        ha = ColavAutomaton()
        assert ha.get_automaton_name() == "COLAV Automaton"

    def test_construction_with_obstacles_and_tuning(self):
        ha = ColavAutomaton(
            waypoint_x=5000.0,
            waypoint_y=600.0,
            obstacles=[(1000.0, 600.0, 5.0, 3.1)],
            Cs=300.0,
            v=5.0,
            tp=3.0,
            K=0.35,
            K_off=0.25,
        )
        assert ha.get_automaton_name() == "COLAV Automaton"

    def test_k_off_default_is_below_k(self):
        # Hysteresis requires K_off < K, otherwise resume can re-trigger
        # avoidance instantly (the T-1022 chattering failure).
        import inspect
        sig = inspect.signature(ColavAutomaton)
        assert sig.parameters['K_off'].default < sig.parameters['K'].default
