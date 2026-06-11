"""
Tests for V1 virtual-waypoint selection (paper Section 4.1 / 4.3).

Ego convention: origin, heading +x, starboard = -y.
"""
import numpy as np

from colav_automaton.controllers.virtual_waypoint import (
    compute_v1,
    default_vertex_provider,
)

CS = 100.0


class TestDefaultVertexProvider:
    def test_no_obstacles(self):
        assert default_vertex_provider(0, 0, [], CS) is None

    def test_eight_vertices_per_obstacle_at_cs(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0)]
        vertices = default_vertex_provider(0, 0, obstacles, CS)
        assert len(vertices) == 8
        for vx, vy in vertices:
            assert abs(np.hypot(vx - 500.0, vy - 0.0) - CS) < 1e-9

    def test_two_obstacles_give_sixteen_vertices(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0), (900.0, 300.0, 0.0, 0.0)]
        assert len(default_vertex_provider(0, 0, obstacles, CS)) == 16


class TestComputeV1:
    def test_no_obstacles(self):
        assert compute_v1(0, 0, 0.0, [], CS, default_vertex_provider) is None

    def test_obstacle_behind_yields_none(self):
        # All candidate vertices sit astern (outside the +/-90 deg window).
        obstacles = [(-500.0, 0.0, 0.0, 0.0)]
        assert compute_v1(0, 0, 0.0, obstacles, CS, default_vertex_provider) is None

    def test_head_on_prefers_starboard(self):
        # Symmetric head-on encounter: port is not >10% better, so the
        # COLREGs starboard default must win (Rules 13-15).
        obstacles = [(500.0, 0.0, 5.0, np.pi)]
        v1 = compute_v1(0, 0, 0.0, obstacles, CS, default_vertex_provider, v=5.0)
        assert v1 is not None
        assert v1[1] < 0.0

    def test_v1_is_ahead(self):
        obstacles = [(500.0, 0.0, 5.0, np.pi)]
        v1 = compute_v1(0, 0, 0.0, obstacles, CS, default_vertex_provider, v=5.0)
        angle = np.arctan2(v1[1] - 0.0, v1[0] - 0.0)
        assert -np.pi / 2 < angle < np.pi / 2

    def test_buffer_moves_v1_no_closer_to_obstacle(self):
        obstacles = [(500.0, 0.0, 5.0, np.pi)]
        v1_plain = compute_v1(0, 0, 0.0, obstacles, CS,
                              default_vertex_provider, v=5.0)
        v1_buffered = compute_v1(0, 0, 0.0, obstacles, CS,
                                 default_vertex_provider,
                                 buffer_distance=25.0, v=5.0)
        d_plain = np.hypot(v1_plain[0] - 500.0, v1_plain[1] - 0.0)
        d_buffered = np.hypot(v1_buffered[0] - 500.0, v1_buffered[1] - 0.0)
        assert d_buffered >= d_plain - 1e-9

    def test_fallback_provider_returning_none_yields_none(self):
        obstacles = [(500.0, 0.0, 0.0, 0.0)]

        def empty_provider(px, py, obs, cs, psi):
            return None

        assert compute_v1(0, 0, 0.0, obstacles, CS, empty_provider) is None
