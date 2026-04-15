"""
Replay a single CommonOcean scenario with visual display (noVNC).

Supports both scenario datasets:
  - handcrafted:      1 ego + dynamic obstacles with trajectories
  - marine_cadastre:  Multiple planning problems, traffic synthesized

Usage inside the Docker container:
    cd /app/commonocean-sim/src

    # Handcrafted (default)
    python3 /app/usv-navigation/commonocean_integration/scripts/replay_scenario.py T-8
    python3 /app/usv-navigation/commonocean_integration/scripts/replay_scenario.py T-27 T-28

    # MarineCadastre
    python3 /app/usv-navigation/commonocean_integration/scripts/replay_scenario.py \\
        --dataset marine_cadastre C-USA_FLO-1_2019011409
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import os

import numpy as np

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
    'handcrafted': {
        'scenarios_dir': '/app/scenarios/HandcraftedTwoVesselEncounters_01_24',
        'Cs': 300.0,
        'max_runtime': 3000,
    },
    'marine_cadastre': {
        'scenarios_dir': '/app/scenarios/MarineCadastre_01_19',
        'Cs': 300.0,
        'max_runtime': 5000,
    },
}


def find_scenario_file(scenarios_dir, scenario_suffix):
    """Find the XML file matching a scenario suffix, searching recursively."""
    # Direct match first
    xml_files = glob.glob(os.path.join(scenarios_dir, '**', '*.xml'), recursive=True)
    for f in xml_files:
        basename = os.path.basename(f).replace('.xml', '')
        if basename == scenario_suffix or basename.endswith(scenario_suffix):
            return f
    return None


def synthesize_traffic_obstacle(pp, ego_t0, scenario_dt, target_dt, obs_id, min_steps=5000):
    """Convert a planning problem into a dynamic obstacle with straight-line trajectory."""
    from commonroad.geometry.shape import Rectangle

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
    shape = Rectangle(length=175.0, width=30.0)
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
    scenario, planning_problem_set = reader.open()

    all_pps = list(planning_problem_set._planning_problem_dict.values())
    ego_pp = all_pps[0]

    if dataset == 'marine_cadastre' and len(all_pps) >= 2:
        # Synthesize traffic from other planning problems
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
        print(f"  {len(all_pps)} vessels: ego PP {ego_pp.planning_problem_id}, "
              f"{len(traffic_pps)} traffic (synthesized)")
    else:
        # Handcrafted / NewYork: use existing dynamic obstacles
        interpolate_dynamic_obstacles(scenario, dt)
        dynamic_obstacles = list(scenario.dynamic_obstacles)
        static_obstacles = list(scenario.static_obstacles)
        print(f"  {len(dynamic_obstacles)} dynamic obstacles, "
              f"{len(static_obstacles)} static obstacles")

    return scenario, ego_pp, dynamic_obstacles, static_obstacles


def run_scenario_visual(scenario_path, dt, config, colav_params, dataset):
    """Run one scenario with the visual displayer enabled."""
    scenario_id = os.path.basename(scenario_path).replace(".xml", "")
    print(f"\n{'='*60}")
    print(f"Replaying: {scenario_id} (dataset: {dataset})")
    print(f"{'='*60}")

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
    goal_rect = {
        'length': goal_shape.length,
        'width': goal_shape.width,
        'orientation': goal_shape.orientation,
    }

    init_state = YPState(
        position=np.array(init.position, dtype="float64"),
        velocity=init.velocity,
        orientation=init.orientation,
        time_step=0,
    )

    scenario_params = {**colav_params, 'v': init.velocity}
    colav_factory = ColavVesselFactory(dt, config, **scenario_params)
    vessel = colav_factory.get_vessel(
        init_state, waypoints, dt,
        ship_name=f"COLAV_{ego_pp.planning_problem_id}",
        vesselid=ego_pp.planning_problem_id,
        goal_waypoint=(goal_pos[0], goal_pos[1]),
    )

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

    for m in sim.models:
        if hasattr(m, 'controller') and hasattr(m.controller, 'sim'):
            m.controller.sim = sim
            m.controller.controlled_vessel = m
            m.controller.real_time_pacing = False
        try:
            m.journey_finished = False
        except AttributeError:
            pass

    stopper = GoalReachedStopper(
        sim.models, goal_pos,
        goal_rect['length'], goal_rect['width'], goal_rect['orientation'],
    )
    stopper.sim = sim
    sim.add_event_listener(stopper)

    collision_flag = [False]
    for listener in sim.listeners:
        if hasattr(listener, 'collision_methods'):
            def _latch(v, o, s, _f=collision_flag):
                _f[0] = True
            listener.collision_methods.append(_latch)

    vessel.controller.shutdown()

    print(f"  Ego: pos=({init.position[0]:.0f}, {init.position[1]:.0f}) "
          f"heading={np.degrees(init.orientation):.1f}deg v={init.velocity:.2f}m/s")
    print(f"  Goal: ({goal_pos[0]:.0f}, {goal_pos[1]:.0f}), dist={approach_dist:.0f}m")
    print(f"  Cs={colav_params['Cs']:.0f}m")
    print(f"  Running with visual display...")

    sim.display_run()

    ctrl = sim.models[0].controller
    print(f"\n  Result: goal={'REACHED' if stopper._stop_next else 'NOT REACHED'}, "
          f"collision={'YES' if collision_flag[0] else 'NO'}, "
          f"steps={ctrl.stepped}")
    ctrl.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Replay scenario with noVNC display")
    parser.add_argument('scenarios', nargs='+',
                        help='Scenario IDs or suffixes (e.g. T-8, C-USA_FLO-1_2019011409)')
    parser.add_argument('--dataset', default='handcrafted',
                        choices=['handcrafted', 'marine_cadastre'],
                        help='Scenario dataset (default: handcrafted)')
    parser.add_argument('--scenarios-dir', default=None,
                        help='Override scenarios directory')
    parser.add_argument('--Cs', type=float, default=None,
                        help='Override safety radius (m)')
    parser.add_argument('--max-runtime', type=int, default=None,
                        help='Override max simulation steps')
    parser.add_argument('--dt', type=float, default=1.0)
    args = parser.parse_args()

    # Apply dataset defaults, then any explicit overrides
    defaults = DATASET_DEFAULTS[args.dataset]
    scenarios_dir = args.scenarios_dir or defaults['scenarios_dir']
    Cs = args.Cs if args.Cs is not None else defaults['Cs']
    max_runtime = args.max_runtime if args.max_runtime is not None else defaults['max_runtime']

    config = setup_config(args.dt, max_runtime)
    config["general_simulator"]["using_displayer"] = True

    colav_params = dict(a=1.67, v=12.0, eta=3.5, tp=3.0, Cs=Cs)

    for suffix in args.scenarios:
        xml_path = find_scenario_file(scenarios_dir, suffix)
        if xml_path is None:
            print(f"ERROR: Could not find scenario matching '{suffix}' in {scenarios_dir}")
            continue
        run_scenario_visual(xml_path, args.dt, config, colav_params, args.dataset)


if __name__ == '__main__':
    main()
