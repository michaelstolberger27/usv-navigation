"""
Adapter bridging the colav_automaton to commonocean-sim's VesselController.

Runs the ColavAutomaton in a background thread with real-time mode enabled,
bridging vessel state from commonocean-sim into the automaton and reading
control outputs back. The automaton handles all guard evaluation, state
transitions, and control computation — identical to standalone usage.

Usage in a VesselFactory::

    controller = HybridAutomatonController(
        vessel, dt,
        a=1.67, v=12.0, eta=3.5, tp=1.0, Cs=2.0,
    )
    vessel.set_controller(controller)
"""

import asyncio
import time
import threading
import numpy as np

from Controller.VesselController import VesselController
from Environment.SurfaceVessel import SurfaceVessel

from colav_automaton import ColavAutomaton
from colav_automaton.controllers import HeadingControlProvider


class HybridAutomatonController(VesselController):
    """
    COLREGs-aware prescribed-time heading controller for commonocean-sim.

    Internally runs a ColavAutomaton in a background asyncio thread with
    real-time pacing.  Each call to control_input() injects the current
    vessel state and reads the latest control output from the automaton.

    Returns [acceleration, yaw_rate] each tick (YP vessel model).
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
        self.sim = None  # set by simulator after construction

        # Shared state — written by control_input(), read by providers
        self._vessel_state = np.array([0.0, 0.0, 0.0])

        # Build automaton exactly like main.py
        self._ha = ColavAutomaton(
            waypoint_x=0.0,
            waypoint_y=0.0,
            obstacles=[],
            Cs=Cs,
            a=a,
            v=v,
            eta=eta,
            tp=tp,
            v1_buffer=v1_buffer,
        )

        self._controller = HeadingControlProvider(self._ha)

        # Background event loop
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait()  # block until activate() is running

        self.stepped = 0
        self.signal_tracker = []
        self.position_tracker = []  # Track actual vessel positions
        self.state_tracker = []  # Track automaton states
        self.v1_tracker = []  # Track virtual waypoints (V1) during avoidance
        self.real_time_pacing = True  # sleep(dt) each tick for display sync
        self._last_waypoint = None  # track waypoint changes
        self._last_automaton_state = None  # detect S2/S3 → S1 transitions

    # ---- background thread ----

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._activate())

    async def _activate(self):
        self._ready.set()
        # Use a fast internal rate so the automaton evaluates control
        # frequently; the adapter's time.sleep() paces the overall sim.
        internal_rate = 20  # Hz
        await self._ha.activate(
            initial_continuous_state=self._vessel_state.copy(),
            initial_control_input_states={'u': np.array([0.0])},
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
        """Inject the latest vessel state into the automaton's context."""
        ctx = self._ha._runtime._ctx
        ctx.continuous_state.set_continuous_state(self._vessel_state.copy())
        return ctx.continuous_state

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

        # 1. Inject vessel state (provider picks it up next cycle)
        self._vessel_state[:] = [pos_x, pos_y, psi]

        # 2. Update waypoint and obstacles in automaton config
        ctx = self._ha._runtime._ctx
        current_wp = (target_state[0], target_state[1])
        if ctx is not None:
            cfg = ctx.configuration
            # Use goal for guard LOS checks (long-range detection).
            # Navigation still steers toward the intermediate waypoint
            # via target_state / yaw_rate computation below.
            if self.goal_waypoint is not None:
                cfg['waypoints'][0] = self.goal_waypoint
            else:
                cfg['waypoints'][0] = current_wp
            cfg['obstacles'] = self._get_obstacles()

            # Reset prescribed-time clock when waypoint changes
            if self._last_waypoint is not None and current_wp != self._last_waypoint:
                ctx.clock.ping_transition()
            self._last_waypoint = current_wp

            # 3. Read latest control output
            u = float(ctx.control_input_states['u'].latest())
            yaw_rate = -self.a * psi + self.a * u
        else:
            # Automaton not yet active — hold heading
            yaw_rate = 0.0

        self.stepped += 1

        # Track actual vessel state from simulation
        self.position_tracker.append([pos_x, pos_y, psi])
        if ctx is not None and hasattr(ctx, 'discrete_state'):
            state_name = ctx.discrete_state.name
            self.state_tracker.append(state_name)

            # Track virtual waypoint V1 when in collision avoidance mode
            # In S2/S3, V1 is at waypoints[-1], goal is deeper in stack
            if state_name in ['COLLISION_AVOIDANCE', 'CONSTANT_CONTROL']:
                if len(cfg['waypoints']) >= 2:
                    v1 = cfg['waypoints'][-1]  # Top of stack is V1
                    self.v1_tracker.append(v1)
                else:
                    self.v1_tracker.append(None)
            else:
                self.v1_tracker.append(None)  # No V1 in S1
        else:
            state_name = 'WAYPOINT_REACHING'
            self.state_tracker.append(state_name)
            self.v1_tracker.append(None)

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

        if self.stepped % 10 == 1:
            u_str = f"{u:.3f}" if ctx is not None else "N/A"
            n_obs = len(cfg['obstacles']) if ctx is not None else 0
            state_name = ctx.discrete_state.name if ctx is not None and hasattr(ctx, 'discrete_state') else "N/A"
            print(f"[step {self.stepped}] pos=({pos_x:.1f},{pos_y:.1f}) "
                  f"psi={np.degrees(psi):.1f}deg "
                  f"wp=({target_state[0]:.0f},{target_state[1]:.0f}) "
                  f"u={u_str} yaw_rate={yaw_rate:.4f} "
                  f"obs={n_obs} state={state_name}")
        signal = np.array([0.0, yaw_rate])
        self.signal_tracker.append(np.copy(signal))

        # Pace the simulator to match the automaton's internal tick
        if self.real_time_pacing:
            time.sleep(1.0 / 20)

        return signal

    def update_after_waypoint(self):
        pass

    def deep_copy(self):
        """Create a fresh controller clone (asyncio loops can't be pickled)."""
        clone = HybridAutomatonController(
            None, self.dt,
            a=self.a, v=self.v, eta=self.eta, tp=self.tp,
            Cs=self.Cs, v1_buffer=self.v1_buffer,
            goal_waypoint=self.goal_waypoint,
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
        """Stop the background automaton loop and release resources."""
        self._ha.deactivate()
        self._thread.join(timeout=2.0)
        # Close the event loop to release file descriptors (selector + self-pipe)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop.close()

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
