#!/usr/bin/env python3
"""
Plot trajectory visualisations for selected CommonOcean scenarios.

Re-runs a handful of scenarios via batch_evaluate.run_single_scenario()
and produces per-scenario trajectory plots showing:
  - Ego vessel path (coloured by automaton state)
  - Traffic vessel path
  - Goal rectangle
  - Start positions and headings
  - CPA marker
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/app/commonocean-sim/src")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D

from commonocean.common.file_reader import CommonOceanFileReader
from Pipeline.SimulationIO import extract_center_point_of_region
from commonocean.scenario.state import YPState
from Simulator.SimulatorFactory import SimulatorFactory

from commonocean_integration.adapters import ColavVesselFactory
from commonocean_integration.sim_utils import (
    interpolate_dynamic_obstacles,
    GoalReachedStopper,
    setup_config,
)

# Reuse the batch infrastructure
from commonocean_integration.scripts.batch_evaluate import run_single_scenario


def plot_scenario(scenario_path, dt, config, colav_params, output_dir):
    """Run one scenario and save a trajectory plot."""
    sid = os.path.basename(scenario_path).replace(".xml", "")
    short = sid.split("T-")[-1]

    # Run scenario
    metrics = run_single_scenario(scenario_path, dt, config, colav_params)

    # Re-load scenario to get traffic trajectory (interpolated to match dt)
    reader = CommonOceanFileReader(scenario_path)
    scenario, pps = reader.open()
    interpolate_dynamic_obstacles(scenario, dt)
    pp = list(pps._planning_problem_dict.values())[0]
    goal_pos = extract_center_point_of_region(pp.goal)
    goal_state = pp.goal.state_list[0]
    goal_shape = goal_state.position
    init = pp.initial_state

    # Traffic trajectory
    dyn_obs = list(scenario.dynamic_obstacles)[0] if scenario.dynamic_obstacles else None
    traffic_positions = []
    if dyn_obs:
        for t in range(3000):
            s = dyn_obs.state_at_time(t)
            if s is None:
                break
            traffic_positions.append(s.position)
        traffic_positions = np.array(traffic_positions)

    # Ego trajectory from metrics (re-run stored in controller)
    # We need to re-run to get position_tracker - run_single_scenario
    # returns metrics dict only.  Instead, parse from the metrics we have.
    # Actually, we need the raw positions.  Let's re-run with a wrapper.
    # For simplicity, just re-run and capture the controller.
    ego_positions = _run_and_get_positions(scenario_path, dt, config, colav_params)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Ego trajectory
    ego_xy = np.array([[p[0], p[1]] for p in ego_positions])
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "b-", linewidth=1.5, alpha=0.8, label="Ego vessel")
    ax.plot(ego_xy[0, 0], ego_xy[0, 1], "bo", markersize=10, zorder=10)
    ax.plot(ego_xy[-1, 0], ego_xy[-1, 1], "bx", markersize=10, zorder=10)

    # Ego start heading arrow
    psi0 = ego_positions[0][2]
    ax.annotate("", xy=(ego_xy[0, 0] + 150 * np.cos(psi0), ego_xy[0, 1] + 150 * np.sin(psi0)),
                xytext=(ego_xy[0, 0], ego_xy[0, 1]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))

    # Traffic trajectory
    if len(traffic_positions) > 0:
        ax.plot(traffic_positions[:, 0], traffic_positions[:, 1],
                "r-", linewidth=1.5, alpha=0.8, label="Traffic vessel")
        ax.plot(traffic_positions[0, 0], traffic_positions[0, 1], "rs", markersize=10, zorder=10)
        # Traffic start heading arrow
        t_psi = dyn_obs.initial_state.orientation
        ax.annotate("", xy=(traffic_positions[0, 0] + 150 * np.cos(t_psi),
                            traffic_positions[0, 1] + 150 * np.sin(t_psi)),
                    xytext=(traffic_positions[0, 0], traffic_positions[0, 1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # CPA marker
    cpa_step = int(metrics["cpa_step"])
    if cpa_step < len(ego_xy) and len(traffic_positions) > cpa_step:
        ax.plot([ego_xy[cpa_step, 0], traffic_positions[cpa_step, 0]],
                [ego_xy[cpa_step, 1], traffic_positions[cpa_step, 1]],
                "k--", linewidth=1, alpha=0.6)
        mid_x = (ego_xy[cpa_step, 0] + traffic_positions[cpa_step, 0]) / 2
        mid_y = (ego_xy[cpa_step, 1] + traffic_positions[cpa_step, 1]) / 2
        ax.annotate(f"CPA={metrics['cpa_distance']:.0f}m",
                    xy=(mid_x, mid_y), fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Goal rectangle
    orient = goal_shape.orientation
    rect = Rectangle(
        (goal_pos[0] - goal_shape.length / 2, goal_pos[1] - goal_shape.width / 2),
        goal_shape.length, goal_shape.width,
        angle=np.degrees(orient),
        linewidth=2, edgecolor="green", facecolor="green", alpha=0.15,
        label="Goal",
    )
    t_rect = Affine2D().rotate_around(goal_pos[0], goal_pos[1], orient) + ax.transData
    rect.set_transform(t_rect)
    ax.add_patch(rect)
    ax.plot(goal_pos[0], goal_pos[1], "g*", markersize=15, zorder=10)

    # Labels
    encounter = metrics.get("encounter_type", "?")
    goal_yn = "Yes" if metrics["goal_reached"] else "No"
    ax.set_title(f"T-{short}  |  {encounter}  |  CPA={metrics['cpa_distance']:.0f}m  |  "
                 f"Goal={goal_yn}  |  Steps={int(metrics['total_steps'])}",
                 fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"trajectory_T-{short}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _run_and_get_positions(scenario_path, dt, config, colav_params):
    """Run a scenario and return the ego position_tracker."""
    reader = CommonOceanFileReader(scenario_path)
    scenario, pps = reader.open()
    interpolate_dynamic_obstacles(scenario, dt)
    pp = list(pps._planning_problem_dict.values())[0]
    init = pp.initial_state

    waypoints = [np.array(init.position, dtype="float64")]
    if pp.waypoints:
        waypoints.extend(pp.generate_reference_points_from_waypoint())
    goal_pos = extract_center_point_of_region(pp.goal)
    waypoints.append(goal_pos)
    approach_dir = goal_pos - waypoints[-2]
    approach_dist = np.linalg.norm(approach_dir)
    if approach_dist > 1e-6:
        overshoot = goal_pos + (approach_dir / approach_dist) * 200.0
    else:
        overshoot = goal_pos + np.array([200.0, 0.0])
    waypoints.append(overshoot)
    waypoints = np.array(waypoints, dtype="float64")

    goal_state = pp.goal.state_list[0]
    goal_shape = goal_state.position
    goal_rect = {
        "length": goal_shape.length,
        "width": goal_shape.width,
        "orientation": goal_shape.orientation,
    }

    init_state = YPState(
        position=np.array(init.position, dtype="float64"),
        velocity=init.velocity,
        orientation=init.orientation,
        time_step=0,
    )

    scenario_params = {**colav_params, "v": init.velocity}
    colav_factory = ColavVesselFactory(dt, config, **scenario_params)
    vessel = colav_factory.get_vessel(
        init_state, waypoints, dt,
        ship_name=f"COLAV_{pp.planning_problem_id}",
        vesselid=pp.planning_problem_id,
        goal_waypoint=(goal_pos[0], goal_pos[1]),
    )

    dynamic_obstacles = list(scenario.dynamic_obstacles)
    static_obstacles = list(scenario.static_obstacles)

    simfac = SimulatorFactory(dt)
    simfac.configure_simulation_factory(
        models=[vessel], dynObsts=dynamic_obstacles, obsts=static_obstacles,
        runnerlist=[], states_rate=1, current_configuration=config,
    )
    sim = simfac.generate_scenario()

    for m in sim.models:
        if hasattr(m, "controller") and hasattr(m.controller, "sim"):
            m.controller.sim = sim
            m.controller.controlled_vessel = m
            m.controller.real_time_pacing = False

    vessel.controller.shutdown()

    stopper = GoalReachedStopper(
        sim.models, goal_pos,
        goal_rect["length"], goal_rect["width"], goal_rect["orientation"],
    )
    stopper.sim = sim
    sim.add_event_listener(stopper)

    sim.display_run()

    ctrl = sim.models[0].controller
    positions = list(ctrl.position_tracker)
    ctrl.shutdown()
    return positions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot scenario trajectories")
    parser.add_argument("--scenarios-dir", default="/app/scenarios")
    parser.add_argument("--output-dir", default="/app/usv-navigation/output/batch_eval/plots")
    parser.add_argument("--ids", nargs="+", type=int, default=[2, 4, 6, 7, 9, 17],
                        help="Scenario T-numbers to plot")
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dt = args.dt
    config = setup_config(dt, max_runtime=3000)
    colav_params = dict(a=1.67, v=12.0, eta=3.5, tp=3.0, Cs=300.0)

    for t_num in args.ids:
        xml_path = os.path.join(args.scenarios_dir, f"ZAM_AAA-1_20240121_T-{t_num}.xml")
        if not os.path.exists(xml_path):
            print(f"Not found: {xml_path}")
            continue
        print(f"Plotting T-{t_num}...")
        try:
            plot_scenario(xml_path, dt, config, colav_params, args.output_dir)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
