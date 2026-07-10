"""
Adapter bridging the colav_automaton to commonocean-sim's VesselController.

Steps a deterministic SyncColavRuntime once per simulation tick: each
control_input() call injects the current vessel state and obstacle list,
advances the automaton by dt of sim time, and applies the returned
control. Guard evaluation, transitions, and control computation are the
same shared implementations the async runtime uses — but evaluated
synchronously, so a scenario replays bit-identically regardless of host
load (the previous background-asyncio design sampled guards at wall-
clock instants, making outcomes run-to-run nondeterministic — one batch
scenario collided in roughly 60% of runs before the migration).

Usage in a VesselFactory::

    controller = HybridAutomatonController(
        vessel, dt,
        a=1.67, v=12.0, eta=3.5, tp=1.0, Cs=2.0,
    )
    vessel.set_controller(controller)
"""

import logging
import time

import numpy as np

from Controller.VesselController import VesselController
from Environment.SurfaceVessel import SurfaceVessel

from colav_automaton import SyncColavRuntime
from colav_automaton.controllers.unsafe_sets import compute_unified_unsafe_region

logger = logging.getLogger(__name__)


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
        # scenarios in a horizon sweep (2026-06-12).
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
        self.v1_side_tracker = []  # 'port'/'starboard' of the active V1, else None
        self.unsafe_set_tracker = []  # Track unsafe set polygon coords each step
        # Display scripts opt in to wall-clock pacing; batch runs leave
        # it off (no effect on the trajectory either way).
        self.real_time_pacing = False
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
            self.v1_side_tracker.append(cfg.get('last_v1_side'))
        else:
            self.v1_tracker.append(None)
            self.v1_side_tracker.append(None)

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
                logger.info("Avoidance recovery: skipped to waypoint index %d", idx)
        self._last_automaton_state = state_name

        # Prevent the sim from terminating on waypoint-path completion.
        # We rely on GoalReachedStopper + max_runtime instead.
        if self.controlled_vessel is not None:
            try:
                self.controlled_vessel.journey_finished = False
            except AttributeError:
                pass

        if self.stepped % 10 == 1:
            logger.debug(
                "[step %d] pos=(%.1f,%.1f) psi=%.1fdeg wp=(%.0f,%.0f) "
                "u=%.3f yaw_rate=%.4f obs=%d state=%s",
                self.stepped, pos_x, pos_y, np.degrees(psi),
                target_state[0], target_state[1], u, yaw_rate,
                len(cfg['obstacles']), state_name)
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

        # Controlled vessels (other than self)
        for model in self.sim.models:
            if model is self.controlled_vessel or model.journey_finished:
                continue
            pos = model.position
            obstacles.append((pos[0], pos[1], model.velocity, model.heading))

        # Dynamic obstacles from scenario
        if self.sim.dynamic_obstacles:
            for dyn_obs in self.sim.dynamic_obstacles:
                state = dyn_obs.state_at_time(self.stepped)
                if state is None:
                    continue
                pos = state.position
                obstacles.append((pos[0], pos[1], state.velocity, state.orientation))

        if self.stepped % 50 == 1:
            logger.debug("%s sees %d obstacles",
                         self.controlled_vessel.vessel_name, len(obstacles))

        return obstacles
