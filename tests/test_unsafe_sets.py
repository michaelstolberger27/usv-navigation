"""
Tests for the unsafe-set API wrappers, including the max_horizon cap on the
swept region (paper eq 27 approximation) used for V1 placement.

Ego convention: origin, heading +x.
"""
import numpy as np
from shapely.geometry import Point, Polygon

from colav_automaton.controllers.unsafe_sets import (
    _compute_swept_obstacles,
    _create_agent,
    _create_obstacles,
    compute_unified_unsafe_region,
    create_los_cone,
    get_unsafe_set_vertices,
)

CS = 300.0


def _sweep_extent(obstacles_list, ship_v, max_horizon):
    """Max displacement of swept samples from the original obstacle."""
    agent = _create_agent(0.0, 0.0, 0.0, ship_v, CS)
    dyn = _create_obstacles(obstacles_list, CS)
    swept = _compute_swept_obstacles(agent, dyn, 0.0, ship_v,
                                     max_horizon=max_horizon)
    ox, oy = obstacles_list[0][0], obstacles_list[0][1]
    return max(np.hypot(o.position[0] - ox, o.position[1] - oy)
               for o in swept)


class TestCreateLosCone:
    def test_contains_midpoint_of_los(self):
        cone = create_los_cone(0, 0, 2000.0, 0.0, 5.0, 60.0)
        assert cone.contains(Point(1000.0, 0.0))

    def test_degenerate_waypoint_at_position(self):
        cone = create_los_cone(0, 0, 0.0, 0.0, 5.0, 60.0)
        assert cone.area > 0.0

    def test_width_scales_with_v_tp(self):
        narrow = create_los_cone(0, 0, 2000.0, 0.0, 5.0, 10.0)
        wide = create_los_cone(0, 0, 2000.0, 0.0, 5.0, 60.0)
        assert wide.area > narrow.area


class TestGetUnsafeSetVertices:
    def test_no_obstacles(self):
        assert get_unsafe_set_vertices(0, 0, [], CS) is None

    def test_static_obstacle_yields_hull(self):
        obstacles = [(800.0, 0.0, 0.0, 0.0)]
        vertices = get_unsafe_set_vertices(0, 0, obstacles, CS, ship_v=5.0)
        assert vertices is not None
        assert len(vertices) >= 3
        assert all(np.isfinite(v).all() for v in np.asarray(vertices))

    def test_max_horizon_caps_swept_extent(self):
        # Head-on approaching obstacle 800 m out: TCPA ~ 800/15 = 53 s.
        # Uncapped, the trajectory samples extend ~53 s * 5 m/s = 267 m;
        # capped at 20 s they must stay within 100 m.
        obstacles = [(800.0, 0.0, 5.0, np.pi)]
        uncapped_extent = _sweep_extent(obstacles, ship_v=10.0,
                                        max_horizon=None)
        capped_extent = _sweep_extent(obstacles, ship_v=10.0,
                                      max_horizon=20.0)
        assert capped_extent <= 20.0 * 5.0 + 1e-6
        assert uncapped_extent > capped_extent

    def test_capped_hull_still_covers_obstacle(self):
        obstacles = [(800.0, 0.0, 5.0, np.pi)]
        capped = get_unsafe_set_vertices(
            0, 0, obstacles, CS, ship_v=10.0,
            use_swept_region=True, max_horizon=20.0)
        hull = Polygon(capped)
        assert hull.contains(Point(800.0, 0.0))


class TestComputeUnifiedUnsafeRegion:
    def test_no_obstacles(self):
        assert compute_unified_unsafe_region(0, 0, [], CS) is None

    def test_returns_polygon_covering_obstacle(self):
        obstacles = [(800.0, 0.0, 0.0, 0.0)]
        region = compute_unified_unsafe_region(0, 0, obstacles, CS, ship_v=5.0)
        assert region is not None
        assert region.contains(Point(800.0, 0.0))

    def test_static_only_is_smaller_for_moving_obstacle(self):
        # static_only drops the velocity-based prediction, which is exactly
        # why G23 uses it for resume checks (full prediction never clears).
        obstacles = [(2000.0, 0.0, 5.0, np.pi)]
        full = compute_unified_unsafe_region(
            0, 0, obstacles, CS, ship_v=10.0, static_only=False)
        static = compute_unified_unsafe_region(
            0, 0, obstacles, CS, ship_v=10.0, static_only=True)
        assert static.area < full.area
