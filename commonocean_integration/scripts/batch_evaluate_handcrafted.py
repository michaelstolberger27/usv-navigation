"""
Batch evaluation of COLAV automaton on HandcraftedTwoVesselEncounters scenarios.

Each scenario has 1 ego vessel (planning problem) and 1 dynamic obstacle (traffic)
with a pre-computed trajectory.

Usage inside the Docker container:
    cd /app/commonocean-sim/src
    python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate_handcrafted.py [OPTIONS]

Options:
    --scenarios-dir PATH   Directory with XML scenario files
                           (default: /app/scenarios/HandcraftedTwoVesselEncounters_01_24)
    --output-dir PATH      Where to write results CSV and plots
                           (default: /app/usv-navigation/output/batch_eval_handcrafted)
    --limit N              Only run first N scenarios (0 = all)
    --start N              Start from scenario index N
    --max-runtime N        Max simulation steps per scenario (default: 3000)
    --resume               Skip already-completed scenarios if CSV exists
    --dt FLOAT             Simulation timestep (default: 1.0)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import glob
import os
import time
import traceback

import numpy as np
import pandas as pd

from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.scenario.state import YPState
from Simulator.SimulatorFactory import SimulatorFactory
from Pipeline.SimulationIO import extract_center_point_of_region

from commonocean_integration.adapters import ColavVesselFactory
from commonocean_integration.evaluation import extract_metrics
from commonocean_integration.sim_utils import (
    interpolate_dynamic_obstacles,
    GoalReachedStopper,
    setup_config,
)


def run_single_scenario(scenario_path, dt, config, colav_params):
    """
    Run one CommonOcean XML scenario with the COLAV controller.

    Returns a dict of metrics, or raises on failure.
    """
    scenario_id = os.path.basename(scenario_path).replace(".xml", "")

    t0 = time.time()

    # Load scenario
    reader = CommonOceanFileReader(scenario_path)
    scenario, planning_problem_set = reader.open()

    # Interpolate traffic trajectories to match sim dt (e.g. 10s → 1s)
    # Also extrapolates to at least 3000 steps so the sim doesn't terminate
    # prematurely when avoidance detours take longer than the original traffic.
    interpolate_dynamic_obstacles(scenario, dt)
    for _dobs in scenario.dynamic_obstacles:
        _pred = _dobs.prediction
        if _pred and _pred.trajectory:
            _tlen = len(_pred.trajectory.state_list)
            print(f"  Traffic trajectory length after interpolation: {_tlen} steps")

    # Get the single planning problem
    pp = list(planning_problem_set._planning_problem_dict.values())[0]

    # Extract waypoints and goal
    # The waypoint list must start with the initial position — commonocean-sim's
    # divide_long_waypointpaths needs at least 2 points to form segments.
    init = pp.initial_state
    waypoints = [np.array(init.position, dtype="float64")]
    if pp.waypoints:
        waypoints.extend(pp.generate_reference_points_from_waypoint())
    goal_pos = extract_center_point_of_region(pp.goal)
    waypoints.append(goal_pos)
    # Add an overshoot waypoint past the goal so the vessel sails *through*
    # the goal rectangle rather than stopping ~87.5m short (the sim's
    # final_waypoint_finished_radius = vessel_length / 2).
    approach_dir = goal_pos - waypoints[-2]
    approach_dist = np.linalg.norm(approach_dir)
    if approach_dist > 1e-6:
        overshoot = goal_pos + (approach_dir / approach_dist) * 2000.0
    else:
        overshoot = goal_pos + np.array([2000.0, 0.0])
    waypoints.append(overshoot)
    waypoints = np.array(waypoints, dtype="float64")

    # Goal rectangle for goal-reached check
    goal_state = pp.goal.state_list[0]
    goal_shape = goal_state.position
    goal_rect = {
        'length': goal_shape.length,
        'width': goal_shape.width,
        'orientation': goal_shape.orientation,
    }

    # Build initial state
    init_state = YPState(
        position=np.array(init.position, dtype="float64"),
        velocity=init.velocity,
        orientation=init.orientation,
        time_step=0,
    )

    # Create COLAV vessel — use the scenario's actual velocity
    scenario_params = {**colav_params, 'v': init.velocity}
    colav_factory = ColavVesselFactory(dt, config, **scenario_params)
    vessel = colav_factory.get_vessel(
        init_state, waypoints, dt,
        ship_name=f"COLAV_{pp.planning_problem_id}",
        vesselid=pp.planning_problem_id,
        goal_waypoint=(goal_pos[0], goal_pos[1]),
    )

    # Dynamic and static obstacles
    dynamic_obstacles = list(scenario.dynamic_obstacles)
    static_obstacles = list(scenario.static_obstacles)

    # Build simulation
    simfac = SimulatorFactory(dt)
    simfac.configure_simulation_factory(
        models=[vessel],
        dynObsts=dynamic_obstacles,
        obsts=static_obstacles,
        runnerlist=[],
        states_rate=1,
        current_configuration=config,
    )

    sim = simfac.generate_scenario()

    # generate_scenario() deep-copies models, so work with sim.models from here
    for m in sim.models:
        if hasattr(m, 'controller') and hasattr(m.controller, 'sim'):
            m.controller.sim = sim
            m.controller.controlled_vessel = m
            m.controller.real_time_pacing = False
        # Prevent the sim from terminating when the vessel thinks its waypoint
        # path is done — we rely on GoalReachedStopper + max_runtime instead.
        try:
            m.journey_finished = False
        except AttributeError:
            pass

    # Stop simulation as soon as the vessel enters the goal rectangle
    stopper = GoalReachedStopper(
        sim.models, goal_pos,
        goal_rect['length'], goal_rect['width'], goal_rect['orientation'],
    )
    stopper.sim = sim
    sim.add_event_listener(stopper)

    # Register a collision flag via the CollisionDetector's collision_methods list.
    # collision_occurred is reset to False at the start of every state_change() call
    # and is never explicitly set to True, so checking it after the run always
    # yields False.  Instead we inject a callback that latches a flag on first hit.
    collision_flag = [False]
    for listener in sim.listeners:
        if hasattr(listener, 'collision_methods'):
            def _latch(v, o, s, _f=collision_flag):
                _f[0] = True
            listener.collision_methods.append(_latch)

    # Shutdown the original (unused) controller's asyncio thread
    vessel.controller.shutdown()

    # Run simulation
    sim.display_run()

    runtime = time.time() - t0

    # Extract metrics from the actual controller that ran
    ctrl = sim.models[0].controller
    metrics = extract_metrics(
        controller=ctrl,
        scenario=scenario,
        goal_pos=np.array(goal_pos),
        goal_rect=goal_rect,
        scenario_id=scenario_id,
        dt=dt,
    )
    metrics['collision_detected'] = collision_flag[0]
    metrics['ego_init_x'] = float(init.position[0])
    metrics['ego_init_y'] = float(init.position[1])
    metrics['ego_init_orientation'] = init.orientation
    metrics['ego_init_velocity'] = init.velocity
    metrics['runtime_sec'] = runtime

    # Shutdown controller (cleanup asyncio thread)
    ctrl.shutdown()

    return metrics


def generate_plots(df, output_dir):
    """Generate summary plots from batch results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_valid = df.dropna(subset=['cpa_distance'])
    df_valid = df_valid[np.isfinite(df_valid['cpa_distance'])]
    if len(df_valid) == 0:
        print("No valid results to plot")
        return

    # 1. CPA Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_valid['cpa_distance'], bins=50, edgecolor='black', alpha=0.7)
    cs_val = 300.0
    ax.axvline(x=cs_val, color='red', linestyle='--', label=f'Cs = {cs_val}m')
    ax.set_xlabel('Closest Point of Approach (m)')
    ax.set_ylabel('Number of Scenarios')
    ax.set_title(f'CPA Distribution (n={len(df_valid)})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpa_distribution.png'), dpi=150)
    plt.close()

    # 2. Collision rate by encounter type
    if 'encounter_type' in df_valid.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = df_valid.groupby('encounter_type').agg(
            total=('collision_detected', 'count'),
            collisions=('collision_detected', 'sum'),
        )
        grouped['collision_rate'] = grouped['collisions'] / grouped['total']
        grouped['collision_rate'].plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_ylabel('Collision Rate')
        ax.set_title('Collision Rate by Encounter Type')
        ax.set_ylim(0, 1)
        for i, (_, row) in enumerate(grouped.iterrows()):
            ax.text(i, row['collision_rate'] + 0.02,
                    f"{row['collisions']:.0f}/{row['total']:.0f}",
                    ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'collision_rate_by_encounter.png'), dpi=150)
        plt.close()

    # 3. State time distribution
    if all(c in df_valid.columns for c in ['time_s1_pct', 'time_s2_pct', 'time_s3_pct']):
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [
            df_valid['time_s1_pct'].mean(),
            df_valid['time_s2_pct'].mean(),
            df_valid['time_s3_pct'].mean(),
        ]
        labels = ['S1: WAYPOINT_REACHING', 'S2: COLLISION_AVOIDANCE', 'S3: CONSTANT_CONTROL']
        colors = ['steelblue', 'coral', 'orange']
        ax.bar(labels, means, color=colors, edgecolor='black')
        ax.set_ylabel('Mean Fraction of Simulation Time')
        ax.set_title('Average Time Spent in Each Automaton State')
        ax.set_ylim(0, 1)
        for i, m in enumerate(means):
            ax.text(i, m + 0.02, f"{m:.1%}", ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_time_distribution.png'), dpi=150)
        plt.close()

    # 4. Goal reached rate
    if 'goal_reached' in df_valid.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        reached = int(df_valid['goal_reached'].sum())
        not_reached = len(df_valid) - reached
        ax.pie([reached, not_reached],
               labels=[f'Reached ({reached})', f'Not Reached ({not_reached})'],
               colors=['seagreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Goal Reached Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goal_reached_rate.png'), dpi=150)
        plt.close()

    # 5. CPA by encounter type (box plot)
    if 'encounter_type' in df_valid.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        encounter_types = sorted(df_valid['encounter_type'].unique())
        data = [df_valid[df_valid['encounter_type'] == et]['cpa_distance'] for et in encounter_types]
        ax.boxplot(data, labels=encounter_types, patch_artist=True)
        ax.axhline(y=cs_val, color='red', linestyle='--', label=f'Cs = {cs_val}m')
        ax.set_ylabel('CPA Distance (m)')
        ax.set_title('CPA by Encounter Type')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cpa_by_encounter.png'), dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch COLAV evaluation")
    parser.add_argument('--scenarios-dir',
                        default='/app/scenarios/HandcraftedTwoVesselEncounters_01_24',
                        help='Directory with XML scenario files')
    parser.add_argument('--output-dir',
                        default='/app/usv-navigation/output/batch_eval_handcrafted',
                        help='Output directory for results')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit to first N scenarios (0 = all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start from scenario index N')
    parser.add_argument('--max-runtime', type=int, default=3000,
                        help='Max simulation steps per scenario')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed scenarios if CSV exists')
    parser.add_argument('--dt', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'results.csv')

    # Discover scenarios sorted by T-number
    xml_files = sorted(
        glob.glob(os.path.join(args.scenarios_dir, '*.xml')),
        key=lambda p: int(os.path.basename(p).split('T-')[1].replace('.xml', '')),
    )
    print(f"Found {len(xml_files)} scenario files")

    # Apply start/limit
    xml_files = xml_files[args.start:]
    if args.limit > 0:
        xml_files = xml_files[:args.limit]
    print(f"Running {len(xml_files)} scenarios (start={args.start}, limit={args.limit})")

    # Resume support
    completed_ids = set()
    existing_results = []
    if args.resume and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        completed_ids = set(df_existing['scenario_id'].tolist())
        existing_results = df_existing.to_dict('records')
        print(f"Resuming: {len(completed_ids)} scenarios already completed")

    config = setup_config(args.dt, args.max_runtime)
    colav_params = dict(a=1.67, v=12.0, eta=3.5, tp=3.0, Cs=300.0)

    results = existing_results
    for i, xml_path in enumerate(xml_files):
        sid = os.path.basename(xml_path).replace(".xml", "")
        if sid in completed_ids:
            continue

        print(f"\n[{i + 1}/{len(xml_files)}] {sid} ...", end=" ", flush=True)
        try:
            metrics = run_single_scenario(xml_path, args.dt, config, colav_params)
            results.append(metrics)
            print(f"OK  steps={metrics['total_steps']}  "
                  f"cpa={metrics['cpa_distance']:.0f}m  "
                  f"goal={'Y' if metrics['goal_reached'] else 'N'}  "
                  f"collision={'Y' if metrics['collision_detected'] else 'N'}  "
                  f"{metrics['runtime_sec']:.1f}s")
        except Exception as e:
            results.append({'scenario_id': sid, 'error': str(e)})
            print(f"FAIL: {e}")
            traceback.print_exc()

        # Free lingering event loops / file descriptors
        gc.collect()

        # Save incrementally every 10 scenarios
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(csv_path, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    valid = df.dropna(subset=['total_steps']) if 'total_steps' in df.columns else df
    collisions = int(valid['collision_detected'].sum()) if 'collision_detected' in valid.columns else '?'
    goals = int(valid['goal_reached'].sum()) if 'goal_reached' in valid.columns else '?'
    print(f"Total: {len(df)} scenarios, {collisions} collisions, {goals} goals reached")

    # Generate summary plots
    generate_plots(df, args.output_dir)


if __name__ == '__main__':
    main()
