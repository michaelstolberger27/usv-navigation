#!/usr/bin/env python3
"""
Animate a CommonOcean scenario as a GIF, similar to examples/output/.

Supports both scenario datasets:
  - handcrafted:      1 ego + dynamic obstacles with trajectories
  - marine_cadastre:  Multiple planning problems, traffic synthesized

Usage inside the Docker container:
    cd /app/commonocean-sim/src

    # Handcrafted (default)
    python3 /app/usv-navigation/commonocean_integration/scripts/animate_scenario.py T-8
    python3 /app/usv-navigation/commonocean_integration/scripts/animate_scenario.py T-27 T-584 --fps 20

    # MarineCadastre
    python3 /app/usv-navigation/commonocean_integration/scripts/animate_scenario.py \
        --dataset marine_cadastre C-USA_FLO-1_2019011409
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/app/commonocean-sim/src")

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.scenario.obstacle import DynamicObstacle, ObstacleType
from commonocean.scenario.state import YPState
from commonocean.scenario.trajectory import Trajectory
from commonocean.prediction.prediction import TrajectoryPrediction
from Simulator.SimulatorFactory import SimulatorFactory
from Pipeline.SimulationIO import extract_center_point_of_region

from commonocean_integration.adapters import ColavVesselFactory
from commonocean_integration.sim_utils import (
    interpolate_dynamic_obstacles,
    GoalReachedStopper,
    setup_config,
)

# Per-dataset defaults
DATASET_DEFAULTS = {
    "handcrafted": {
        "scenarios_dir": "/app/scenarios/HandcraftedTwoVesselEncounters_01_24",
        "output_dir": "/app/usv-navigation/output/batch_eval_handcrafted/animations",
        "Cs": 300.0,
        "max_runtime": 3000,
    },
    "marine_cadastre": {
        "scenarios_dir": "/app/scenarios/MarineCadastre_01_19",
        "output_dir": "/app/usv-navigation/output/batch_eval_marine_cadastre/animations",
        "Cs": 300.0,
        "max_runtime": 5000,
    },
}


def find_scenario_file(scenarios_dir, scenario_suffix):
    """Find the XML file matching a scenario suffix, searching recursively."""
    xml_files = glob.glob(os.path.join(scenarios_dir, "**", "*.xml"), recursive=True)
    for f in xml_files:
        basename = os.path.basename(f).replace(".xml", "")
        if basename == scenario_suffix or basename.endswith(scenario_suffix):
            return f
    return None


def synthesize_traffic_obstacle(pp, ego_t0, scenario_dt, target_dt, obs_id, min_steps=5000):
    """Convert a planning problem into a dynamic obstacle with straight-line trajectory."""
    from commonroad.geometry.shape import Rectangle as CRRectangle

    init = pp.initial_state
    v = init.velocity
    heading = init.orientation
    px, py = init.position

    time_offset_scenario = init.time_step - ego_t0
    time_offset_steps = int(round(time_offset_scenario * scenario_dt / target_dt))

    if time_offset_steps != 0:
        dt_shift = -time_offset_steps * target_dt
        px += v * np.cos(heading) * dt_shift
        py += v * np.sin(heading) * dt_shift

    state_list = []
    for t in range(1, min_steps + 1):
        state_list.append(YPState(
            time_step=t,
            position=np.array([
                px + v * np.cos(heading) * t * target_dt,
                py + v * np.sin(heading) * t * target_dt,
            ]),
            velocity=v,
            orientation=heading,
        ))

    init_state = YPState(
        time_step=0,
        position=np.array([px, py]),
        velocity=v,
        orientation=heading,
    )

    trajectory = Trajectory(initial_time_step=1, state_list=state_list)
    shape = CRRectangle(length=175.0, width=30.0)
    prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)

    return DynamicObstacle(
        obstacle_id=obs_id,
        obstacle_type=ObstacleType.MOTORVESSEL,
        obstacle_shape=shape,
        initial_state=init_state,
        prediction=prediction,
    )


def load_scenario(scenario_path, dt, dataset):
    """
    Load a scenario and return (scenario, ego_pp, dynamic_obstacles, static_obstacles).

    For marine_cadastre, synthesizes traffic from other planning problems.
    For handcrafted/new_york, uses existing dynamic obstacles.
    """
    reader = CommonOceanFileReader(scenario_path)
    scenario, pps = reader.open()

    all_pps = list(pps._planning_problem_dict.values())
    ego_pp = all_pps[0]

    if dataset == "marine_cadastre" and len(all_pps) >= 2:
        traffic_pps = all_pps[1:]
        scenario_dt = scenario.dt or 10.0
        ego_t0 = ego_pp.initial_state.time_step

        dynamic_obstacles = []
        for i, tpp in enumerate(traffic_pps):
            obs = synthesize_traffic_obstacle(
                tpp, ego_t0, scenario_dt, dt, obs_id=1000 + i)
            dynamic_obstacles.append(obs)
            scenario._dynamic_obstacles[obs.obstacle_id] = obs

        static_obstacles = list(scenario.static_obstacles)
    else:
        interpolate_dynamic_obstacles(scenario, dt)
        dynamic_obstacles = list(scenario.dynamic_obstacles)
        static_obstacles = list(scenario.static_obstacles)

    return scenario, ego_pp, dynamic_obstacles, static_obstacles


def run_scenario(scenario_path, dt, config, colav_params, dataset):
    """
    Run a scenario and return all data needed for animation.

    Returns dict with keys:
        ego_positions, ego_states, ego_v1s, traffic_positions_list,
        goal_pos, goal_shape, init, collision, goal_reached, total_steps, Cs
    """
    scenario, ego_pp, dynamic_obstacles, static_obstacles = \
        load_scenario(scenario_path, dt, dataset)

    init = ego_pp.initial_state

    waypoints = [np.array(init.position, dtype="float64")]
    if ego_pp.waypoints:
        waypoints.extend(ego_pp.generate_reference_points_from_waypoint())
    goal_pos = extract_center_point_of_region(ego_pp.goal)
    waypoints.append(goal_pos)
    approach_dir = goal_pos - waypoints[-2]
    approach_dist = np.linalg.norm(approach_dir)
    if approach_dist > 1e-6:
        overshoot = goal_pos + (approach_dir / approach_dist) * 2000.0
    else:
        overshoot = goal_pos + np.array([2000.0, 0.0])
    waypoints.append(overshoot)
    waypoints = np.array(waypoints, dtype="float64")

    goal_state = ego_pp.goal.state_list[0]
    goal_shape = goal_state.position

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
        ship_name=f"COLAV_{ego_pp.planning_problem_id}",
        vesselid=ego_pp.planning_problem_id,
        goal_waypoint=(goal_pos[0], goal_pos[1]),
    )

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
        try:
            m.journey_finished = False
        except AttributeError:
            pass

    vessel.controller.shutdown()

    stopper = GoalReachedStopper(
        sim.models, goal_pos,
        goal_shape.length, goal_shape.width, goal_shape.orientation,
    )
    stopper.sim = sim
    sim.add_event_listener(stopper)

    collision_flag = [False]
    for listener in sim.listeners:
        if hasattr(listener, "collision_methods"):
            def _latch(v, o, s, _f=collision_flag):
                _f[0] = True
            listener.collision_methods.append(_latch)

    sim.display_run()

    ctrl = sim.models[0].controller

    # Get traffic trajectories for ALL dynamic obstacles
    traffic_positions_list = []
    for dyn_obs in dynamic_obstacles:
        positions = []
        for t in range(ctrl.stepped + 1):
            s = dyn_obs.state_at_time(t)
            if s is None:
                break
            positions.append(s.position)
        if positions:
            traffic_positions_list.append(np.array(positions))

    result = {
        "ego_positions": list(ctrl.position_tracker),
        "ego_states": list(ctrl.state_tracker),
        "ego_v1s": list(ctrl.v1_tracker),
        "traffic_positions_list": traffic_positions_list,
        "goal_pos": goal_pos,
        "goal_shape": goal_shape,
        "init": init,
        "collision": collision_flag[0],
        "goal_reached": stopper._stop_next,
        "total_steps": ctrl.stepped,
        "Cs": colav_params["Cs"],
    }

    ctrl.shutdown()
    return result


def animate_scenario(data, scenario_id, output_path, fps=15, max_frames=450):
    """Create an animated GIF from simulation data."""
    ego_pos = np.array([[p[0], p[1]] for p in data["ego_positions"]])
    ego_psi = np.array([p[2] for p in data["ego_positions"]])
    ego_states = data["ego_states"]
    ego_v1s = data["ego_v1s"]
    traffic_list = data["traffic_positions_list"]
    goal_pos = data["goal_pos"]
    goal_shape = data["goal_shape"]
    Cs = data["Cs"]
    n_steps = len(ego_pos)

    # Subsample frames
    if n_steps > max_frames:
        step = n_steps / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        if indices[-1] != n_steps - 1:
            indices.append(n_steps - 1)
    else:
        indices = list(range(n_steps))

    # Compute plot bounds
    all_x = list(ego_pos[:, 0])
    all_y = list(ego_pos[:, 1])
    for tpos in traffic_list:
        all_x.extend(tpos[:, 0])
        all_y.extend(tpos[:, 1])
    all_x.append(goal_pos[0])
    all_y.append(goal_pos[1])
    margin = max(Cs * 0.5, 200)
    xlim = (min(all_x) - margin, max(all_x) + margin)
    ylim = (min(all_y) - margin, max(all_y) + margin)

    # State colours
    state_colors = {
        "WAYPOINT_REACHING": "royalblue",
        "COLLISION_AVOIDANCE": "red",
        "CONSTANT_CONTROL": "orange",
    }
    state_labels = {
        "WAYPOINT_REACHING": "S1: Waypoint Reaching",
        "COLLISION_AVOIDANCE": "S2: Collision Avoidance",
        "CONSTANT_CONTROL": "S3: Constant Control",
    }

    # Traffic colours for multiple obstacles
    traffic_colors = ["red", "darkred", "crimson", "orangered", "firebrick",
                      "indianred", "tomato", "salmon"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Goal rectangle
    orient = goal_shape.orientation
    rect = Rectangle(
        (goal_pos[0] - goal_shape.length / 2, goal_pos[1] - goal_shape.width / 2),
        goal_shape.length, goal_shape.width,
        linewidth=2, edgecolor="green", facecolor="green", alpha=0.15,
    )
    t_rect = Affine2D().rotate_around(goal_pos[0], goal_pos[1], orient) + ax.transData
    rect.set_transform(t_rect)
    ax.add_patch(rect)
    ax.plot(goal_pos[0], goal_pos[1], "g*", markersize=18, zorder=20, label="Goal")

    # Static elements
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)

    # Create dynamic elements for each traffic vessel
    traffic_trails = []
    traffic_markers = []
    safety_circles = []
    for i in range(len(traffic_list)):
        c = traffic_colors[i % len(traffic_colors)]
        trail, = ax.plot([], [], "-", color=c, linewidth=1.5, alpha=0.6)
        marker, = ax.plot([], [], "s", color=c, markersize=9, zorder=15)
        traffic_trails.append(trail)
        traffic_markers.append(marker)
        safety_circles.append(None)

    # Ego elements
    ego_trail, = ax.plot([], [], "-", color="royalblue", linewidth=1.5, alpha=0.6)
    ego_marker, = ax.plot([], [], "o", color="royalblue", markersize=10, zorder=15)
    ego_arrow_patch = [None]
    v1_marker, = ax.plot([], [], "m^", markersize=10, zorder=18)
    v1_line, = ax.plot([], [], "m--", linewidth=1, alpha=0.5)
    cpa_line, = ax.plot([], [], "k--", linewidth=1, alpha=0.5)
    cpa_text = ax.text(0, 0, "", fontsize=8, ha="center",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                       visible=False, zorder=25)

    state_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=13,
        verticalalignment="top", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    time_text = ax.text(
        0.02, 0.90, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue",
               markersize=10, label="Ego vessel"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markersize=9, label="Traffic vessel"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="green",
               markersize=14, label="Goal"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="magenta",
               markersize=10, label="V1 waypoint"),
        Line2D([0], [0], color="royalblue", linewidth=2, label="S1: Waypoint"),
        Line2D([0], [0], color="red", linewidth=2, label="S2: Avoidance"),
        Line2D([0], [0], color="orange", linewidth=2, label="S3: Constant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Track CPA across all traffic
    min_cpa = [float("inf")]
    min_cpa_step = [0]
    min_cpa_traffic_idx = [0]

    def update(frame):
        idx = indices[frame]
        x, y = ego_pos[idx]
        psi = ego_psi[idx]
        state = ego_states[idx] if idx < len(ego_states) else "WAYPOINT_REACHING"

        # Ego trail
        trail_x = ego_pos[:idx + 1, 0]
        trail_y = ego_pos[:idx + 1, 1]
        color = state_colors.get(state, "royalblue")
        ego_trail.set_data(trail_x, trail_y)
        ego_trail.set_color(color)
        ego_marker.set_data([x], [y])
        ego_marker.set_color(color)

        # Heading arrow
        if ego_arrow_patch[0] is not None:
            ego_arrow_patch[0].remove()
        arrow_len = max((xlim[1] - xlim[0]) * 0.03, 50)
        ego_arrow_patch[0] = FancyArrow(
            x, y, arrow_len * np.cos(psi), arrow_len * np.sin(psi),
            width=arrow_len * 0.25, head_width=arrow_len * 0.5,
            head_length=arrow_len * 0.3,
            fc=color, ec="black", linewidth=0.5, zorder=16,
        )
        ax.add_patch(ego_arrow_patch[0])

        # Traffic vessels
        for ti, tpos in enumerate(traffic_list):
            if idx < len(tpos):
                tx, ty = tpos[idx]
                traffic_trails[ti].set_data(tpos[:idx + 1, 0], tpos[:idx + 1, 1])
                traffic_markers[ti].set_data([tx], [ty])

                # Safety circle around closest traffic only
                dist = np.hypot(x - tx, y - ty)
                if dist < min_cpa[0]:
                    min_cpa[0] = dist
                    min_cpa_step[0] = idx
                    min_cpa_traffic_idx[0] = ti
            else:
                traffic_markers[ti].set_data([], [])

        # Safety circle on the closest traffic vessel at current step
        for ti in range(len(traffic_list)):
            if safety_circles[ti] is not None:
                safety_circles[ti].remove()
                safety_circles[ti] = None
        if traffic_list:
            # Find closest at current step
            closest_ti = 0
            closest_dist = float("inf")
            for ti, tpos in enumerate(traffic_list):
                if idx < len(tpos):
                    d = np.hypot(x - tpos[idx][0], y - tpos[idx][1])
                    if d < closest_dist:
                        closest_dist = d
                        closest_ti = ti
            if closest_dist < float("inf") and idx < len(traffic_list[closest_ti]):
                tx, ty = traffic_list[closest_ti][idx]
                safety_circles[closest_ti] = Circle(
                    (tx, ty), Cs, color="red", alpha=0.08, fill=True, zorder=3,
                )
                ax.add_patch(safety_circles[closest_ti])

        # CPA line
        ci = min_cpa_step[0]
        cti = min_cpa_traffic_idx[0]
        if ci <= idx and ci < len(traffic_list[cti]) if traffic_list else False:
            ex, ey = ego_pos[ci]
            cx, cy = traffic_list[cti][ci]
            cpa_line.set_data([ex, cx], [ey, cy])
            mid_x = (ex + cx) / 2
            mid_y = (ey + cy) / 2
            cpa_text.set_position((mid_x, mid_y))
            cpa_text.set_text(f"CPA={min_cpa[0]:.0f}m")
            cpa_text.set_visible(True)
        else:
            cpa_line.set_data([], [])
            cpa_text.set_visible(False)

        # V1 waypoint
        v1 = ego_v1s[idx] if idx < len(ego_v1s) else None
        if v1 is not None:
            v1_marker.set_data([v1[0]], [v1[1]])
            v1_line.set_data([x, v1[0]], [y, v1[1]])
        else:
            v1_marker.set_data([], [])
            v1_line.set_data([], [])

        # Text overlays
        state_text.set_text(state_labels.get(state, state))
        state_text.set_color(state_colors.get(state, "black"))
        time_text.set_text(f"t = {idx}s  |  step {idx}/{n_steps}")

        title = scenario_id
        if idx == indices[-1]:
            result_parts = []
            if data["goal_reached"]:
                result_parts.append("GOAL REACHED")
            if data["collision"]:
                result_parts.append("COLLISION")
            if result_parts:
                title += f"  |  {', '.join(result_parts)}"
        ax.set_title(title, fontsize=13, fontweight="bold")

        return []

    anim = FuncAnimation(
        fig, update, frames=len(indices),
        interval=int(1000 / fps), blit=False, repeat=False,
    )

    print(f"  Saving {len(indices)} frames at {fps}fps to {output_path}")
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=72)
    plt.close(fig)
    print(f"  Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Animate CommonOcean scenarios as GIFs"
    )
    parser.add_argument(
        "scenarios", nargs="+",
        help="Scenario IDs or suffixes (e.g. T-8, C-USA_FLO-1_2019011409)",
    )
    parser.add_argument("--dataset", default="handcrafted",
                        choices=["handcrafted", "marine_cadastre"],
                        help="Scenario dataset (default: handcrafted)")
    parser.add_argument("--scenarios-dir", default=None,
                        help="Override scenarios directory")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--Cs", type=float, default=None,
                        help="Override safety radius (m)")
    parser.add_argument("--max-runtime", type=int, default=None,
                        help="Override max simulation steps")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    # Apply dataset defaults, then any explicit overrides
    defaults = DATASET_DEFAULTS[args.dataset]
    scenarios_dir = args.scenarios_dir or defaults["scenarios_dir"]
    output_dir = args.output_dir or defaults["output_dir"]
    Cs = args.Cs if args.Cs is not None else defaults["Cs"]
    max_runtime = args.max_runtime if args.max_runtime is not None else defaults["max_runtime"]

    os.makedirs(output_dir, exist_ok=True)
    config = setup_config(args.dt, max_runtime)
    colav_params = dict(a=1.67, v=12.0, eta=3.5, tp=3.0, Cs=Cs)

    for suffix in args.scenarios:
        xml_path = find_scenario_file(scenarios_dir, suffix)
        if xml_path is None:
            # For handcrafted, also try normalising "8" → "T-8"
            if args.dataset == "handcrafted" and not suffix.upper().startswith("T-"):
                xml_path = find_scenario_file(scenarios_dir, f"T-{suffix}")
            if xml_path is None:
                print(f"ERROR: No scenario matching '{suffix}' in {scenarios_dir}")
                continue

        scenario_id = os.path.basename(xml_path).replace(".xml", "")
        print(f"\nProcessing {scenario_id} (dataset: {args.dataset})...")

        try:
            data = run_scenario(xml_path, args.dt, config, colav_params, args.dataset)
            out_path = os.path.join(output_dir, f"{scenario_id}.gif")
            animate_scenario(
                data, scenario_id, out_path,
                fps=args.fps, max_frames=args.max_frames,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
