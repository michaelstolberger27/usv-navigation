"""
Behavioural regression suite for the COLAV automaton on SyncColavRuntime.

Two halves:

1. Canonical COLREGs encounters (head-on, crossing give-way, overtaking,
   no-obstacle control) run end to end at the CommonOcean-validated scale
   (Cs=300 m, v=6 m/s, tp=3 s, dt=1 s). Each pins the transition sequence,
   the COLREGs-expected turn direction, a minimum separation >= Cs, and
   goal arrival. The runtime is tick-synchronous and the traffic is
   propagated by closed-form constant-velocity kinematics from fixed
   constants, so every run is bit-identical — the asserted numbers are
   exact behaviour pins, not tolerances.

2. Known-defect pins: the two dense-traffic failures documented under
   "Known limitations" in the README, reproduced deterministically as
   strict xfails. They assert the *correct* behaviour, so the day a fix
   lands they flip to XPASS and strict=True fails the suite — the marker
   removal is then the "defect resolved" commit.

All scenario geometries and asserted values were validated empirically
against the current runtime before being pinned (2026-07-05).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pytest

from colav_automaton import SyncColavRuntime
from colav_automaton.guards.conditions import G11_check, G23_check

# CommonOcean-validated parameter scale (same as tests/test_sync_runtime.py).
CS = 300.0
V = 6.0
TP = 3.0
DT = 1.0
PARAMS = dict(Cs=CS, v=V, tp=TP)

S1 = "WAYPOINT_REACHING"


def propagate(obstacles0, t: float):
    """Constant-velocity traffic at sim time t (closed form: deterministic)."""
    return [
        (ox + ov * np.cos(op) * t, oy + ov * np.sin(op) * t, ov, op)
        for ox, oy, ov, op in obstacles0
    ]


@dataclass
class EncounterLog:
    traj: np.ndarray                      # (n, 3) [x, y, psi] per tick
    transitions: List[str]                # fired transition names, in order
    v1_sides: List[Optional[str]]         # cfg['last_v1_side'] at each 'avoid'
    min_sep: List[float]                  # per-obstacle min separation over the run
    goal_reached: bool
    final_mode: str = S1
    modes: List[str] = field(default_factory=list)


def run_encounter(
    waypoint: Tuple[float, float],
    obstacles0,
    initial_state=(0.0, 0.0, 0.0),
    max_steps: int = 1500,
) -> EncounterLog:
    """Drive SyncColavRuntime.step through a scripted encounter."""
    rt = SyncColavRuntime(waypoint=waypoint, obstacles=list(obstacles0),
                          initial_state=initial_state, **PARAMS)
    traj, transitions, v1_sides, modes = [], [], [], []
    min_sep = [np.inf] * len(obstacles0)
    for k in range(max_steps):
        obs = propagate(obstacles0, k * DT)
        r = rt.step(DT, obstacles=obs)
        traj.append(r.state.copy())
        modes.append(r.mode)
        if r.transition:
            transitions.append(r.transition)
            if r.transition == "avoid":
                v1_sides.append(rt.configuration.get('last_v1_side'))
        x, y, _ = r.state
        for i, (ox, oy, _, _) in enumerate(obs):
            min_sep[i] = min(min_sep[i], float(np.hypot(ox - x, oy - y)))
        if rt.goal_reached():
            break
    return EncounterLog(
        traj=np.array(traj),
        transitions=transitions,
        v1_sides=v1_sides,
        min_sep=min_sep,
        goal_reached=rt.goal_reached(),
        final_mode=modes[-1],
        modes=modes,
    )


class TestCanonicalEncounters:
    def test_no_obstacle_direct_to_goal(self):
        # Control case: initial heading offset exercises the prescribed-time
        # convergence, but no guard may ever fire.
        log = run_encounter((3000.0, 400.0), [],
                            initial_state=(0.0, 0.0, 0.5), max_steps=800)
        assert log.transitions == []
        assert set(log.modes) == {S1}
        assert log.goal_reached

    def test_head_on_gives_way_to_starboard(self):
        # Rule 14: reciprocal traffic near the track line — both vessels
        # alter to starboard. Geometry is the sync_runtime docstring
        # example (80 m port offset; the exact-centreline variant clears
        # at only ~244 m < Cs and is deliberately not pinned here).
        log = run_encounter((4000.0, 0.0), [(1500.0, 80.0, 5.0, np.pi)],
                            max_steps=1000)
        assert log.transitions == ["avoid", "hold", "resume"]
        assert log.final_mode == S1
        assert log.goal_reached
        assert min(log.min_sep) >= CS          # observed: 352.1 m
        # Starboard both as decided (V1 side) and as flown (cross-track).
        assert log.v1_sides == ["starboard"]
        assert log.traj[:, 1].min() < -100.0   # observed: -279 m excursion

    def test_crossing_from_starboard_gives_way_to_starboard(self):
        # Rule 15: traffic crossing from starboard, ego is give-way and
        # must not cross ahead. Collision timing by construction: both
        # vessels reach (1800, 0) at t = 300 s.
        log = run_encounter((5000.0, 0.0), [(1800.0, -1500.0, 5.0, np.pi / 2)],
                            max_steps=1300)
        assert log.transitions == ["avoid", "hold", "resume"]
        assert log.final_mode == S1
        assert log.goal_reached
        assert min(log.min_sep) >= CS          # observed: 1010.3 m
        assert log.v1_sides == ["starboard"]
        assert log.traj[:, 1].min() < -100.0   # passes astern, starboard side

    def test_overtaking_clears_slow_vessel(self):
        # Rule 13: slow same-course vessel ahead. Either side is
        # permitted when overtaking, so no side assertion — only
        # clearance and completion. NOTE: overtaking clearance is
        # structurally close to Cs at this scale (the pass happens while
        # cutting back toward track); this geometry clears at 312 m, a
        # 4% margin. A failure here means the safety floor regressed.
        log = run_encounter((5000.0, 0.0), [(900.0, 40.0, 1.5, 0.0)],
                            max_steps=1200)
        assert log.transitions == ["avoid", "hold", "resume"]
        assert log.final_mode == S1
        assert log.goal_reached
        assert min(log.min_sep) >= CS          # observed: 312.0 m

    def test_encounters_are_deterministic(self):
        # Guards the suite's core assumption: exact-value pins are safe
        # because identical inputs give bit-identical runs.
        a = run_encounter((4000.0, 0.0), [(1500.0, 80.0, 5.0, np.pi)],
                          max_steps=1000)
        b = run_encounter((4000.0, 0.0), [(1500.0, 80.0, 5.0, np.pi)],
                          max_steps=1000)
        assert np.array_equal(a.traj, b.traj)
        assert a.transitions == b.transitions
        assert a.min_sep == b.min_sep


# ---------------------------------------------------------------------------
# Known-defect pins (see "Known limitations" in README.md). Each asserts the
# CORRECT behaviour and xfails strictly today: when a fix lands, the test
# XPASSes, strict=True turns that into a suite failure, and removing the
# marker becomes part of the fix commit.
# ---------------------------------------------------------------------------

# Two traffic lanes flanking an empty corridor. Every vessel is 1200 m
# (= 4*Cs) laterally from the ego->goal LOS segment, and every heading is
# lane-parallel, so no current or TCPA-predicted position ever approaches
# the corridor.
_LANE_TRAFFIC = (
    [(float(x), 1200.0, 4.0, np.pi) for x in range(200, 4001, 600)]     # westbound
    + [(float(x), -1200.0, 4.0, 0.0) for x in range(200, 4001, 600)]    # eastbound
)


class TestKnownDefects:
    @pytest.mark.xfail(
        strict=True,
        reason="Known limitation (README): compute_unified_unsafe_region "
               "builds ONE convex hull over all obstacles; the hull of two "
               "flanking lanes covers the empty corridor between them, so G11 "
               "fires (and G23 never clears) with no vessel anywhere near the "
               "LOS. Fix direction: per-obstacle regions / union-not-hull.",
    )
    def test_flanking_lanes_do_not_block_clear_corridor(self):
        # Precondition (true today and after any fix): the corridor is
        # genuinely clear. The LOS segment (0,0)->(4000,0) lies on the
        # x-axis and spans every vessel's x, so lateral clearance is
        # simply |y|; require > cone half-width (Cs) + safety radius (Cs)
        # with 2*Cs to spare.
        for ox, oy, _, _ in _LANE_TRAFFIC:
            assert 0.0 <= ox <= 4000.0
            assert abs(oy) >= 4 * CS

        # Correct behaviour: neither the entry guard geometry (G11) nor
        # the resume guard geometry (G23) sees a blocked LOS. Today both
        # return True because the unified hull spans the corridor.
        assert not G11_check(0.0, 0.0, 0.0, 4000.0, 0.0, V, TP,
                             _LANE_TRAFFIC, CS)
        assert not G23_check(0.0, 0.0, 0.0, 4000.0, 0.0, V, TP,
                             _LANE_TRAFFIC, CS)

    @pytest.mark.xfail(
        strict=True,
        reason="Known limitation (README): S3 resume requires the GLOBAL max "
               "risk index < K_off; a steady stream of distant traffic keeps "
               "RI >= K_off forever and the ego freezes in S3 after the real "
               "threat has passed. Fix direction: per-threat hysteresis "
               "(resume when the RI of the obstacle(s) that triggered "
               "avoidance subsides).",
    )
    def test_resume_after_threat_passes_despite_background_traffic(self):
        # One genuine crossing threat from starboard (meets the track at
        # (1200, 0) at t = 200 s) triggers avoidance, crosses to the
        # north half-plane and recedes. A conveyor of 12 westbound
        # vessels at y = +1500 (spaced 1200 m, so at every tick some
        # vessel has TCPA <= 120 s toward the ego) keeps the GLOBAL max
        # risk index at ~1/3 >= K_off = 0.25 through the TCPA term alone
        # — while never coming within 3*Cs of the ego, and staying (with
        # all post-crossing traffic) in the north half-plane so the
        # static G23 hull never touches the LOS cone.
        #
        # Instrumented reproduction (2026-07-05): hold fires at t = 274 s
        # and the ego then spends 627/627 remaining ticks in S3 with
        # G23_check False and RI in [0.33, 0.48] >= K_off — the resume
        # block is purely the global-K_off term, not geometry. The held
        # heading misses the goal by ~390 m.
        threat = [(1200.0, -1000.0, 5.0, np.pi / 2)]
        conveyor = [(1000.0 + 1200.0 * i, 1500.0, 5.0, np.pi)
                    for i in range(12)]
        log = run_encounter((4000.0, 0.0), threat + conveyor, max_steps=900)

        # Preconditions (true today and after any fix): avoidance was a
        # real, single activation, and the conveyor never became an
        # actual threat.
        assert log.transitions[:2] == ["avoid", "hold"]
        assert min(log.min_sep[1:]) >= 2 * CS   # conveyor observed >= 933.7 m

        # Correct behaviour: once the crossing vessel has passed, the
        # ship resumes waypoint reaching and makes the goal. Today it
        # stays frozen in S3 for the rest of the run.
        assert "resume" in log.transitions
        assert log.goal_reached
