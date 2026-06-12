"""
Adapter bridging the colav_automaton to commonocean-sim's VesselController.

Steps a deterministic SyncColavRuntime once per simulation tick: each
control_input() call injects the current vessel state and obstacle list,
advances the automaton by dt of sim time, and applies the returned
control. Guard evaluation, transitions, and control computation are the
same shared implementations the async runtime uses — but evaluated
synchronously, so a scenario replays bit-identically regardless of host
load (the previous background-asyncio design sampled guards at wall-
clock instants, making outcomes run-to-run nondeterministic — see
HANDOFF on T-838).

Usage in a VesselFactory::

    controller = HybridAutomatonController(
        vessel, dt,
        a=1.67, v=12.0, eta=3.5, tp=1.0, Cs=2.0,
    )
    vessel.set_controller(controller)
"""

import time
import numpy as np

from Controller.VesselController import VesselController
from Environment.SurfaceVessel import SurfaceVessel

from colav_automaton import SyncColavRuntime
from colav_automaton.controllers.unsafe_sets import compute_unified_unsafe_region


class HybridAutomatonController(VesselController):
    """
    COLREGs-aware prescribed-time heading controller for commonocean-sim.

    Steps a SyncColavRuntime once per control_input() call (one sim
    tick). Returns [acceleration, yaw_rate] each tick (YP vessel model).
    """

    def __init__(
        self,
        vessel: SurfaceVessel,
        dt: float,
        *,
        a: float,
        v: float,
        eta: float,
        tp: float,
        Cs: float,
        v1_buffer: float = 0.0,
        goal_waypoint: tuple = None,
        tp_control: float = None,
    ):
        super().__init__(vessel)
        self.dt = dt
        self.a = a
        self.v = v
        self.eta = eta
        self.tp = tp
        self.Cs = Cs
        self.v1_buffer = v1_buffer
        self.goal_waypoint = goal_waypoint  # final goal for long-range G11 check
        # Prescribed-time control horizon: the law's singular gain ramp
        # must span many evaluation steps (paper assumes dt << tp). A
        # literal tp of a few seconds at dt=1 destabilises the YP plant;
        # 60*dt reproduced the async-validated CPAs across the key
        # scenarios and fixed T-838 (sweep 2026-06-12, HANDOFF §2).
        self.tp_control = tp_control if tp_control is not None else max(60.0 * dt, tp)
        self.sim = None  # set by simulator after construction

        # Deterministic tick-synchronous automaton. The waypoint is
        # rewritten each tick (goal_waypoint or the sim's current
        # target), exactly as the previous adapter did.
        self._rt = SyncColavRuntime(
            waypoint=(0.0, 0.0),
            obstacles=[],
            initial_state=(0.0, 0.0, 0.0),
            Cs=Cs, a=a, v=v, eta=eta, tp=tp, v1_buffer=v1_buffer,
            tp_control=self.tp_control,
        )

        self.stepped = 0
        self.signal_tracker = []
        self.position_tracker = []  # Track actual vessel positions
        self.state_tracker = []  # Track automaton states
        self.v1_tracker = []  # Track virtual waypoints (V1) during avoidance
        self.unsafe_set_tracker = []  # Track unsafe set polygon coords each step
        self.real_time_pacing = True  # sleep(dt) each tick for display sync
        self._last_waypoint = None  # track waypoint changes
        self._last_automaton_state = None  # detect S2/S3 → S1 transitions

    # ---- VesselController interface ----

    def control_input(self, target_state: np.ndarray, current_state) -> np.ndarray:
        """
        Compute one-step control signal.

        Called each simulation tick by the commonocean-sim Simulator.

        Returns:
            np.ndarray [acceleration, yaw_rate] for the YP vessel model.
        """
        pos_x, pos_y = current_state.position
        psi = current_state.orientation

        cfg = self._rt.configuration
        current_wp = (target_state[0], target_state[1])

        # Use goal for guard LOS checks (long-range detection).
        # Navigation still steers toward the automaton's waypoint stack
        # top (goal in S1, V1 in S2/S3).
        if self.goal_waypoint is not None:
            cfg['waypoints'][0] = self.goal_waypoint
        else:
            cfg['waypoints'][0] = current_wp

        # Reset prescribed-time clock when the sim's waypoint changes
        if self._last_waypoint is not None and current_wp != self._last_waypoint:
            self._rt.notify_waypoint_changed()
        self._last_waypoint = current_wp

        # One deterministic automaton tick (state in, control out;
        # the sim's vessel model owns integration)
        result = self._rt.step_external(
            self.dt, [pos_x, pos_y, psi], obstacles=self._get_obstacles()
        )
        u = result.u
        yaw_rate = -self.a * psi + self.a * u
        state_name = result.mode

        self.stepped += 1

        # Track actual vessel state from simulation
        self.position_tracker.append([pos_x, pos_y, psi])
        self.state_tracker.append(state_name)

        # Track virtual waypoint V1 when in collision avoidance mode
        # In S2/S3, V1 is at waypoints[-1], goal is deeper in stack
        if state_name in ['COLLISION_AVOIDANCE', 'CONSTANT_CONTROL'] \
                and len(cfg['waypoints']) >= 2:
            self.v1_tracker.append(cfg['waypoints'][-1])
        else:
            self.v1_tracker.append(None)

        # Track unsafe set polygon for visualisation (best-effort, skip on error)
        try:
            obstacles = cfg['obstacles']
            if obstacles:
                poly = compute_unified_unsafe_region(
                    pos_x, pos_y, obstacles, self.Cs,
                    ship_psi=psi, ship_v=self.v
                )
                if poly is not None:
                    coords = list(poly.exterior.coords)
                    self.unsafe_set_tracker.append(coords)
                else:
                    self.unsafe_set_tracker.append(None)
            else:
                self.unsafe_set_tracker.append(None)
        except Exception:
            self.unsafe_set_tracker.append(None)

        # After avoidance (S2/S3 → S1), skip past intermediate waypoints
        # that are now behind the vessel so it doesn't backtrack.
        if (state_name == 'WAYPOINT_REACHING'
                and self._last_automaton_state is not None
                and self._last_automaton_state != 'WAYPOINT_REACHING'
                and self.controlled_vessel is not None):
            vessel = self.controlled_vessel
            if hasattr(vessel, 'waypoints') and vessel.waypoints is not None:
                wps = vessel.waypoints
                idx = vessel.next_waypoint_index
                while idx < len(wps) - 1:
                    wp_dir = wps[idx] - np.array([pos_x, pos_y])
                    fwd = np.array([np.cos(psi), np.sin(psi)])
                    # Skip waypoint if it's behind the vessel
                    if np.dot(wp_dir, fwd) < 0:
                        vessel.passed_waypoint = wps[idx]
                        idx += 1
                        vessel.next_waypoint_index = idx
                        vessel.next_waypoint = wps[idx]
                    else:
                        break
                print(f"  [AVOIDANCE RECOVERY] Skipped to waypoint index {idx}")
        self._last_automaton_state = state_name

        # Prevent the sim from terminating on waypoint-path completion.
        # We rely on GoalReachedStopper + max_runtime instead.
        if self.controlled_vessel is not None:
            try:
                self.controlled_vessel.journey_finished = False
            except AttributeError:
                pass

        if self.stepped % 10 == 1:
            print(f"[step {self.stepped}] pos=({pos_x:.1f},{pos_y:.1f}) "
                  f"psi={np.degrees(psi):.1f}deg "
                  f"wp=({target_state[0]:.0f},{target_state[1]:.0f}) "
                  f"u={u:.3f} yaw_rate={yaw_rate:.4f} "
                  f"obs={len(cfg['obstacles'])} state={state_name}")
        signal = np.array([0.0, yaw_rate])
        self.signal_tracker.append(np.copy(signal))

        # Pace the simulator for display sync (no effect on the
        # trajectory — the automaton ticks in sim time either way)
        if self.real_time_pacing:
            time.sleep(1.0 / 20)

        return signal

    def update_after_waypoint(self):
        pass

    def deep_copy(self):
        """Create a fresh controller clone."""
        clone = HybridAutomatonController(
            None, self.dt,
            a=self.a, v=self.v, eta=self.eta, tp=self.tp,
            Cs=self.Cs, v1_buffer=self.v1_buffer,
            goal_waypoint=self.goal_waypoint,
            tp_control=self.tp_control,
        )
        clone.controlled_vessel = None
        return clone

    def equals(self, controller) -> bool:
        if not isinstance(controller, HybridAutomatonController):
            return False
        return (
            self.a == controller.a
            and self.dt == controller.dt
        )

    def initialise(self):
        pass

    def shutdown(self):
        """Nothing to stop — the sync runtime has no background loop."""
        pass

    # ---- obstacle access ----

    def _get_obstacles(self):
        if self.sim is None:
            return []

        obstacles = []

        if self.stepped % 50 == 1:  # Debug every 50 steps
            n_models = len(self.sim.models)
            n_dyn = len(self.sim.dynamic_obstacles) if self.sim.dynamic_obstacles else 0
            print(f"  [{self.controlled_vessel.vessel_name}] Scanning {n_models} models + {n_dyn} dynamic obstacles")

        # Controlled vessels (other than self)
        for model in self.sim.models:
            if model is self.controlled_vessel:
                if self.stepped % 50 == 1:
                    print(f"    - Skipping self: {model.vessel_name}")
                continue
            if model.journey_finished:
                if self.stepped % 50 == 1:
                    print(f"    - Skipping finished: {model.vessel_name}")
                continue

            pos = model.position
            speed = model.velocity
            heading = model.heading

            if self.stepped % 50 == 1:
                print(f"    - Model: {model.vessel_name} at ({pos[0]:.1f}, {pos[1]:.1f}), "
                      f"speed={speed:.1f} m/s, heading={np.degrees(heading):.1f}°")

            obstacles.append((pos[0], pos[1], speed, heading))

        # Dynamic obstacles from scenario
        if self.sim.dynamic_obstacles:
            for dyn_obs in self.sim.dynamic_obstacles:
                # Get current state at this timestep
                state = dyn_obs.state_at_time(self.stepped)
                if state is None:
                    continue

                pos = state.position
                speed = state.velocity
                heading = state.orientation

                if self.stepped % 50 == 1:
                    print(f"    - DynObs {dyn_obs.obstacle_id}: at ({pos[0]:.1f}, {pos[1]:.1f}), "
                          f"speed={speed:.1f} m/s, heading={np.degrees(heading):.1f}°")

                obstacles.append((pos[0], pos[1], speed, heading))

        if self.stepped % 50 == 1:
            print(f"  [{self.controlled_vessel.vessel_name}] Total obstacles: {len(obstacles)}")

        return obstacles
