"""
Shared utilities for CommonOcean simulation integration.

Contains functions used across multiple scripts:
- interpolate_dynamic_obstacles: trajectory interpolation for dt mismatch
- GoalReachedStopper: event listener that stops sim when goal is reached
- setup_config: load and configure commonocean-sim for headless runs
"""

import numpy as np

from commonocean.scenario.trajectory import Trajectory
from commonocean.prediction.prediction import TrajectoryPrediction
from Simulator.EventListener import EventListener
from rules.common.helper import load_yaml


def interpolate_dynamic_obstacles(scenario, target_dt):
    """Interpolate dynamic obstacle trajectories to match the simulation dt.

    CommonOcean scenarios define trajectories at scenario.dt intervals (e.g.
    10s).  When the simulation runs at a finer dt (e.g. 1s), the simulator
    indexes traffic via state_at_time(sim_step), causing a time-scale
    mismatch.  This function linearly interpolates each trajectory so that
    consecutive integer time steps correspond to target_dt seconds.
    """
    scenario_dt = scenario.dt
    if scenario_dt is None or abs(scenario_dt - target_dt) < 1e-6:
        return  # already matching

    factor = int(round(scenario_dt / target_dt))
    if factor <= 1:
        return

    for dyn_obs in scenario.dynamic_obstacles:
        pred = dyn_obs.prediction
        if pred is None or not isinstance(pred, TrajectoryPrediction):
            continue

        old_states = pred.trajectory.state_list
        if not old_states:
            continue

        # Build full state list including initial_state for interpolation
        init = dyn_obs.initial_state
        all_states = [init] + list(old_states)

        old_ts = np.array([s.time_step for s in all_states], dtype=float)
        pos_x = np.array([s.position[0] for s in all_states])
        pos_y = np.array([s.position[1] for s in all_states])
        vel = np.array([s.velocity for s in all_states])
        orient = np.unwrap([s.orientation for s in all_states])

        # New fine-grained time steps (e.g. 0,1,2,...,1700 instead of 0,1,...,170)
        new_ts = np.arange(
            old_ts[0] * factor,
            (old_ts[-1] + 1) * factor,
            1,
            dtype=float,
        )

        # Map old integer time steps to the new scale
        old_ts_scaled = old_ts * factor

        new_px = np.interp(new_ts, old_ts_scaled, pos_x)
        new_py = np.interp(new_ts, old_ts_scaled, pos_y)
        new_vel = np.interp(new_ts, old_ts_scaled, vel)
        new_orient = np.interp(new_ts, old_ts_scaled, orient)

        StateClass = type(init)

        # Build trajectory states (skip index 0 = initial state)
        new_state_list = []
        for i in range(1, len(new_ts)):
            new_state_list.append(StateClass(**{
                'time_step': int(new_ts[i]),
                'position': np.array([new_px[i], new_py[i]]),
                'velocity': float(new_vel[i]),
                'orientation': float(new_orient[i] % (2 * np.pi)),
            }))

        new_traj = Trajectory(
            initial_time_step=new_state_list[0].time_step,
            state_list=new_state_list,
        )
        dyn_obs._prediction = TrajectoryPrediction(
            trajectory=new_traj,
            shape=pred.shape,
        )


class GoalReachedStopper(EventListener):
    """Stop the simulation once the ego vessel enters the goal rectangle.

    Uses a one-step delay so that control_input() records the
    goal-entering position before the sim loop exits.
    """

    def __init__(self, models, goal_center, goal_length, goal_width, goal_orientation):
        super().__init__(models)
        self.goal_center = goal_center
        self.half_l = goal_length / 2.0
        self.half_w = goal_width / 2.0
        self.c = np.cos(-goal_orientation)
        self.s = np.sin(-goal_orientation)
        self.sim = None
        self._stop_next = False

    def state_change(self, time: float):
        if self._stop_next:
            if self.sim is not None:
                self.sim.is_running = False
            return

        vessel = self.models[0]
        dx = vessel.position[0] - self.goal_center[0]
        dy = vessel.position[1] - self.goal_center[1]
        local_x = self.c * dx - self.s * dy
        local_y = self.s * dx + self.c * dy
        if abs(local_x) <= self.half_l and abs(local_y) <= self.half_w:
            self._stop_next = True

    def remove_vessel(self, vessel_id):
        pass


def setup_config(dt, max_runtime):
    """Load and configure commonocean-sim for headless batch runs."""
    config = load_yaml("/app/commonocean-sim/src/configuration.yaml")
    config["general_simulator"]["dt"] = dt
    config["general_simulator"]["using_collision_avoider"] = False
    config["general_simulator"]["using_collision_detection"] = True
    config["general_simulator"]["using_displayer"] = False
    config["general_simulator"]["maximum_runtime"] = max_runtime
    config["general_simulator"]["plotting"]["do_plotting"] = False
    return config
