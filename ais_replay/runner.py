"""
Drives the COLAV automaton through AIS traffic.

Uses the same pattern as commonocean_integration/adapters/controller.py:
the automaton runs in a background asyncio thread in real-time mode with
a continuous-state provider; the replay loop injects the ego state and
the tracker's obstacle list each tick, reads the control input u back,
and integrates simple heading/speed kinematics externally.

Operating regime: like the CommonOcean batch scripts, the replay loop
runs unpaced (CPU-bound, typically tens of sim-seconds per wall-second)
while the automaton thread evaluates at 20 Hz wall — control input is
therefore updated every few sim-seconds. This is the regime the
2000-scenario batch validation ran in. The ego model clamps yaw rate
(as the sim's vessel model does); without that clamp the sparse control
updates destabilise the heading integration.

Inherited caveat (HANDOFF §3.2): wall-clock evaluation makes exact
trajectories load-dependent. The roadmap's phase 4 (tick-synchronous
runtime) is the proper fix.
"""

import asyncio
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from colav_automaton import ColavAutomaton
from colav_automaton.controllers import HeadingControlProvider
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
        max_yaw_rate: float = 0.15,
        pace: float = 0.02,
        max_duration: float = 7200.0,
        obstacle_range: float = 8000.0,
    ):
        """
        Args:
            max_yaw_rate: ego yaw-rate saturation (rad/s). Stands in for
                the vessel model's physical limit; required for stable
                integration with sparse control updates.
            pace: wall seconds to sleep per tick. The default (0.02)
                holds the wall:sim ratio near the regime the CommonOcean
                batch validation ran in (~50 ticks/s against the 20 Hz
                automaton thread = control refresh every ~2.5 sim s).
                Use dt for 1:1 live/display sync; 0 (flat out) starves
                the automaton of evaluations and is only useful for
                smoke tests.
            obstacle_range: only tracks within this range of the ego are
                passed to the automaton (a strait bbox holds 100+
                vessels; per-obstacle hull computation cannot take them
                all every tick). 0 disables filtering.
        """
        self.source = source
        self.tracker = tracker
        self.goal = goal
        self.v = v
        self.Cs = Cs
        self.dt = dt
        self.a = a
        self.max_yaw_rate = max_yaw_rate
        self.pace = pace
        self.max_duration = max_duration
        self.obstacle_range = obstacle_range

        self._vessel_state = np.array(ego_start, dtype=float)

        self._ha = ColavAutomaton(
            waypoint_x=goal[0],
            waypoint_y=goal[1],
            obstacles=[],
            Cs=Cs, a=a, v=v, eta=eta, tp=tp, K=K, K_off=K_off,
        )
        self._controller = HeadingControlProvider(self._ha)

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

        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    # ---- automaton background thread (adapter pattern) ----

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._activate())

    async def _activate(self):
        self._ready.set()
        internal_rate = 20  # Hz
        await self._ha.activate(
            initial_continuous_state=self._vessel_state.copy(),
            initial_control_input_states={'u': np.array([self._vessel_state[2]])},
            enable_real_time_mode=True,
            enable_self_integration=False,
            delta_time=1.0 / internal_rate,
            timeout_sec=float('inf'),
            continuous_state_provider=self._state_provider,
            continuous_state_provision_rate=internal_rate,
            control_states_provider=self._controller,
            control_states_provision_rate=internal_rate,
            should_write_logs=False,
        )

    def _state_provider(self):
        ctx = self._ha._runtime._ctx
        ctx.continuous_state.set_continuous_state(self._vessel_state.copy())
        return ctx.continuous_state

    # ---- replay loop ----

    def run(self, t_start: Optional[float] = None, verbose: bool = True) -> dict:
        """
        Run the replay. Returns a summary dict (goal_reached, min_cpa,
        steps, sim duration).
        """
        self._thread.start()
        self._ready.wait()
        time.sleep(0.5)  # let activate() reach its first evaluation

        sim_t = self.source.t_start if t_start is None else t_start
        t0 = sim_t
        arrival_radius = max(5.0, self.v * 3.0 * 0.5)

        while sim_t - t0 < self.max_duration:
            # 1. Advance traffic to sim time
            self.source.feed_until(self.tracker, sim_t)
            ego_xy = (self._vessel_state[0], self._vessel_state[1])
            obstacles = self.tracker.obstacles_at(
                sim_t, near=ego_xy, within=self.obstacle_range)

            # 2. Inject ego state + world into the automaton
            ctx = self._ha._runtime._ctx
            state_name = "WAYPOINT_REACHING"
            if ctx is not None:
                cfg = ctx.configuration
                cfg['obstacles'] = obstacles
                u = float(ctx.control_input_states['u'].latest())
                if hasattr(ctx, 'discrete_state') and ctx.discrete_state is not None:
                    state_name = getattr(ctx.discrete_state, 'name', state_name)
            else:
                u = self._vessel_state[2]

            # 3. Integrate ego kinematics (heading control + const speed).
            # yaw_rate = a*(u - psi) with the difference wrapped (the +psi
            # term in u cancels, so this is wrap-safe), saturated at the
            # vessel's physical limit.
            x, y, psi = self._vessel_state
            err = np.arctan2(np.sin(u - psi), np.cos(u - psi))
            yaw_rate = float(np.clip(self.a * err,
                                     -self.max_yaw_rate, self.max_yaw_rate))
            psi = psi + yaw_rate * self.dt
            psi = float(np.arctan2(np.sin(psi), np.cos(psi)))
            x = x + self.v * np.cos(psi) * self.dt
            y = y + self.v * np.sin(psi) * self.dt
            self._vessel_state[:] = [x, y, psi]

            # 4. Record (all parallel trackers, every tick)
            self.times.append(sim_t - t0)
            self.position_tracker.append([x, y, psi])
            self.state_tracker.append(state_name)
            self.traffic_tracker.append(obstacles)
            v1 = None
            if ctx is not None and state_name == "COLLISION_AVOIDANCE":
                wps = ctx.configuration.get('waypoints', [])
                if len(wps) > 1:
                    v1 = tuple(wps[-1])
            self.v1_tracker.append(v1)
            try:
                poly = compute_unified_unsafe_region(
                    x, y, obstacles, self.Cs, ship_psi=psi, ship_v=self.v
                ) if obstacles else None
                self.unsafe_set_tracker.append(
                    list(poly.exterior.coords) if poly is not None else None)
            except Exception:
                self.unsafe_set_tracker.append(None)

            # 5. Metrics + termination
            for ox, oy, _, _ in obstacles:
                self.min_cpa = min(self.min_cpa, float(np.hypot(ox - x, oy - y)))
            if np.hypot(self.goal[0] - x, self.goal[1] - y) < arrival_radius:
                self.goal_reached = True
                break

            sim_t += self.dt
            if self.pace > 0:
                time.sleep(self.pace)

        self._ha.deactivate()
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
                  f"min_cpa={cpa:.0f} m" if cpa is not None else "no traffic met")
        return summary
