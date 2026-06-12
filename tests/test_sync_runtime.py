"""
Tests for the deterministic tick-synchronous runtime.

The determinism tests are the point of the whole class: identical
inputs must give bit-identical trajectories and transition timings,
which the wall-clock async runtime cannot guarantee (HANDOFF §3).
"""
import numpy as np
import pytest

from colav_automaton import SyncColavRuntime

PARAMS = dict(Cs=300.0, v=6.0, tp=3.0)


def head_on_run(n_max=1500, dt=1.0):
    """Scripted head-on encounter; returns (trajectory, transitions, cpa)."""
    rt = SyncColavRuntime(
        waypoint=(5000.0, 0.0),
        obstacles=[(2500.0, 0.0, 5.0, np.pi)],
        initial_state=(0.0, 0.0, 0.0),
        **PARAMS,
    )
    traj, transitions, min_cpa = [], [], np.inf
    for k in range(n_max):
        ox = 2500.0 - 5.0 * (k * dt)
        r = rt.step(dt, obstacles=[(ox, 0.0, 5.0, np.pi)])
        traj.append(r.state)
        if r.transition:
            transitions.append((r.t, r.transition))
        min_cpa = min(min_cpa, float(np.hypot(r.state[0] - ox, r.state[1])))
        if rt.goal_reached():
            break
    return np.array(traj), transitions, min_cpa


class TestDeterminism:
    def test_bit_identical_reruns(self):
        traj1, trans1, cpa1 = head_on_run()
        traj2, trans2, cpa2 = head_on_run()
        assert np.array_equal(traj1, traj2)   # exact, not approx
        assert trans1 == trans2
        assert cpa1 == cpa2

    def test_no_state_leaks_between_instances(self):
        # A first runtime's V1 pushes must not contaminate a second's
        # waypoint stack (cfg dicts must be per-instance).
        _ = head_on_run()
        rt = SyncColavRuntime(waypoint=(100.0, 0.0), **PARAMS)
        assert rt.configuration['waypoints'] == [(100.0, 0.0)]


class TestAutomatonBehaviour:
    def test_full_avoidance_cycle(self):
        traj, transitions, min_cpa = head_on_run()
        names = [name for _, name in transitions]
        assert names[:3] == ["avoid", "hold", "resume"]

    def test_head_on_avoided_and_goal_reached(self):
        traj, transitions, min_cpa = head_on_run()
        assert len(traj) < 1500          # reached goal before cutoff
        assert min_cpa > 150.0           # never closer than half Cs
        # COLREGs: head-on avoidance goes starboard (negative y)
        assert traj[:, 1].min() < -50.0

    def test_no_obstacles_goes_straight_to_goal(self):
        rt = SyncColavRuntime(waypoint=(2000.0, 0.0),
                              initial_state=(0.0, 0.0, 0.0), **PARAMS)
        for _ in range(600):
            r = rt.step(1.0)
            assert r.mode == "WAYPOINT_REACHING"
            if rt.goal_reached():
                break
        assert rt.goal_reached()

    def test_transition_resets_prescribed_time_clock(self):
        rt = SyncColavRuntime(
            waypoint=(5000.0, 0.0),
            obstacles=[(2500.0, 0.0, 5.0, np.pi)],
            initial_state=(0.0, 0.0, 0.0),
            **PARAMS,
        )
        for k in range(200):
            ox = 2500.0 - 5.0 * k
            r = rt.step(1.0, obstacles=[(ox, 0.0, 5.0, np.pi)])
            if r.transition == "avoid":
                assert rt.t - rt._t_last_transition == 0.0
                return
        pytest.fail("avoid transition never fired")

    def test_v1_pushed_and_popped(self):
        rt = SyncColavRuntime(
            waypoint=(5000.0, 0.0),
            obstacles=[(2500.0, 0.0, 5.0, np.pi)],
            initial_state=(0.0, 0.0, 0.0),
            **PARAMS,
        )
        saw_v1 = False
        for k in range(1500):
            ox = 2500.0 - 5.0 * k
            r = rt.step(1.0, obstacles=[(ox, 0.0, 5.0, np.pi)])
            wps = rt.configuration['waypoints']
            if r.mode == "COLLISION_AVOIDANCE":
                assert len(wps) == 2     # goal + V1
                saw_v1 = True
            elif r.mode == "WAYPOINT_REACHING":
                assert len(wps) == 1     # V1 popped on S2 exit
            if rt.goal_reached():
                break
        assert saw_v1

    def test_goal_reached_radius(self):
        rt = SyncColavRuntime(waypoint=(10.0, 0.0),
                              initial_state=(0.0, 0.0, 0.0), **PARAMS)
        assert rt.goal_reached(radius=20.0)
        assert not rt.goal_reached(radius=5.0)

    def test_notify_waypoint_changed_resets_clock(self):
        rt = SyncColavRuntime(waypoint=(2000.0, 0.0),
                              initial_state=(0.0, 0.0, 0.0), **PARAMS)
        for _ in range(10):
            rt.step(1.0)
        assert rt.t - rt._t_last_transition == 10.0
        rt.notify_waypoint_changed()
        assert rt.t - rt._t_last_transition == 0.0


class TestStepExternal:
    """Host-owned integration (the CommonOcean adapter pattern)."""

    def _drive(self, n_max=1500, dt=1.0):
        rt = SyncColavRuntime(
            waypoint=(5000.0, 0.0),
            obstacles=[(2500.0, 0.0, 5.0, np.pi)],
            initial_state=(0.0, 0.0, 0.0),
            **PARAMS,
        )
        # The host integrates the same heading plant the flows use
        x, y, psi = 0.0, 0.0, 0.0
        a, v = 1.67, PARAMS['v']
        traj, transitions = [], []
        for k in range(n_max):
            ox = 2500.0 - 5.0 * (k * dt)
            r = rt.step_external(dt, [x, y, psi],
                                 obstacles=[(ox, 0.0, 5.0, np.pi)])
            psi += (-a * psi + a * r.u) * dt
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            traj.append((x, y, psi))
            if r.transition:
                transitions.append((r.t, r.transition))
            if np.hypot(5000.0 - x, y) < 10.0:
                break
        return np.array(traj), transitions

    def test_external_integration_runs_full_cycle(self):
        traj, transitions = self._drive()
        names = [n for _, n in transitions]
        assert names[:3] == ["avoid", "hold", "resume"]
        assert len(traj) < 1500  # reached goal

    def test_external_is_deterministic(self):
        t1, tr1 = self._drive()
        t2, tr2 = self._drive()
        assert np.array_equal(t1, t2)
        assert tr1 == tr2

    def test_exit_reset_overwrites_u(self):
        # apply_exit_avoidance sets u = psi on S3->S1 so the host's
        # first post-resume yaw_rate is zero (see its docstring); the
        # returned u must reflect the post-reset buffer.
        rt = SyncColavRuntime(
            waypoint=(5000.0, 0.0),
            obstacles=[(2500.0, 0.0, 5.0, np.pi)],
            initial_state=(0.0, 0.0, 0.0),
            **PARAMS,
        )
        x, y, psi = 0.0, 0.0, 0.0
        a, v = 1.67, PARAMS['v']
        for k in range(1500):
            ox = 2500.0 - 5.0 * k
            r = rt.step_external(1.0, [x, y, psi],
                                 obstacles=[(ox, 0.0, 5.0, np.pi)])
            if r.transition == "resume":
                assert r.u == pytest.approx(psi)
                return
            psi += (-a * psi + a * r.u) * 1.0
            x += v * np.cos(psi) * 1.0
            y += v * np.sin(psi) * 1.0
        pytest.fail("resume transition never fired")
