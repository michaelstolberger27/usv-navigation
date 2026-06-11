"""
Drives the COLAV automaton through AIS traffic.

Built on SyncColavRuntime: the automaton is stepped tick-by-tick in sim
time, so replays are deterministic (bit-identical reruns), run as fast
as the CPU allows, and need none of the wall-clock workarounds the
async adapter pattern required (yaw-rate clamping against stale control,
pacing to hold a 'validated wall:sim ratio'). Control is recomputed
every tick by construction.

The optional `pace` parameter exists only for watching a replay in real
time — it has no effect on the trajectory.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from colav_automaton import SyncColavRuntime
from colav_automaton.controllers.unsafe_sets import compute_unified_unsafe_region

from ais_replay.sources import RecordedAISSource
from ais_replay.tracker import TrafficTracker


class ReplayRunner:
    """Run the ego automaton through a recorded AIS scenario."""

    def __init__(
        self,
        source: RecordedAISSource,
        tracker: TrafficTracker,
        ego_start: Tuple[float, float, float],   # x, y, psi (local frame)
        goal: Tuple[float, float],               # x, y (local frame)
        *,
        v: float = 6.0,
        Cs: float = 300.0,
        tp: float = 3.0,
        a: float = 1.67,
        eta: float = 3.5,
        K: float = 0.35,
        K_off: float = 0.25,
        dt: float = 1.0,
        pace: float = 0.0,
        max_duration: float = 7200.0,
        obstacle_range: float = 8000.0,
    ):
        """
        Args:
            pace: wall seconds to sleep per tick, for real-time viewing
                only (0 = run flat out; deterministic either way).
            obstacle_range: only tracks within this range of the ego are
                passed to the automaton (a strait bbox holds 100+
                vessels; per-obstacle hull computation cannot take them
                all every tick). 0 disables filtering.
        """
        self.source = source
        self.tracker = tracker
        self.goal = goal
        self.Cs = Cs
        self.v = v
        self.dt = dt
        self.pace = pace
        self.max_duration = max_duration
        self.obstacle_range = obstacle_range

        self._rt = SyncColavRuntime(
            waypoint=goal,
            obstacles=[],
            initial_state=ego_start,
            Cs=Cs, a=a, v=v, eta=eta, tp=tp, K=K, K_off=K_off,
        )

        # Step-indexed parallel trackers (same convention as the
        # CommonOcean adapter — keep them length-aligned).
        self.times: List[float] = []
        self.position_tracker: List[List[float]] = []
        self.state_tracker: List[str] = []
        self.v1_tracker: List[Optional[Tuple[float, float]]] = []
        self.unsafe_set_tracker: List[Optional[list]] = []
        self.traffic_tracker: List[list] = []

        self.goal_reached = False
        self.min_cpa = float("inf")

    def run(self, t_start: Optional[float] = None, verbose: bool = True) -> dict:
        """
        Run the replay. Returns a summary dict (goal_reached, min_cpa,
        steps, sim duration).
        """
        sim_t = self.source.t_start if t_start is None else t_start
        t0 = sim_t
        rt = self._rt

        while sim_t - t0 < self.max_duration:
            # 1. Advance traffic to sim time
            self.source.feed_until(self.tracker, sim_t)
            x, y, _ = rt.state
            obstacles = self.tracker.obstacles_at(
                sim_t, near=(x, y), within=self.obstacle_range)

            # 2. One deterministic automaton tick
            result = rt.step(self.dt, obstacles=obstacles)
            x, y, psi = result.state

            # 3. Record (all parallel trackers, every tick)
            self.times.append(sim_t - t0)
            self.position_tracker.append([x, y, psi])
            self.state_tracker.append(result.mode)
            self.traffic_tracker.append(obstacles)
            wps = rt.configuration['waypoints']
            self.v1_tracker.append(
                tuple(wps[-1])
                if result.mode == "COLLISION_AVOIDANCE" and len(wps) > 1
                else None)
            try:
                poly = compute_unified_unsafe_region(
                    x, y, obstacles, self.Cs, ship_psi=psi, ship_v=self.v
                ) if obstacles else None
                self.unsafe_set_tracker.append(
                    list(poly.exterior.coords) if poly is not None else None)
            except Exception:
                self.unsafe_set_tracker.append(None)

            # 4. Metrics + termination
            for ox, oy, _, _ in obstacles:
                self.min_cpa = min(self.min_cpa, float(np.hypot(ox - x, oy - y)))
            if rt.goal_reached():
                self.goal_reached = True
                break

            sim_t += self.dt
            if self.pace > 0:
                time.sleep(self.pace)

        summary = {
            "goal_reached": self.goal_reached,
            "min_cpa": None if self.min_cpa == float("inf") else self.min_cpa,
            "steps": len(self.times),
            "sim_duration": self.times[-1] if self.times else 0.0,
        }
        if verbose:
            cpa = summary["min_cpa"]
            print(f"Replay done: steps={summary['steps']} "
                  f"goal={'Y' if self.goal_reached else 'N'} "
                  + (f"min_cpa={cpa:.0f} m" if cpa is not None else "no traffic met"))
        return summary
