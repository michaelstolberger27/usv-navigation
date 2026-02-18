"""
Run a commonocean-sim XML scenario with the COLAV automaton controller.

Usage inside the Docker container:
    cd /app/commonocean-sim/src
    python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py [scenario.xml]
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rules.common.helper import load_yaml
from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.scenario.state import YPState
from Simulator.SimulatorFactory import SimulatorFactory
from Pipeline.SimulationIO import extract_center_point_of_region

from commonocean_integration.adapters import ColavVesselFactory
from commonocean_integration.sim_utils import interpolate_dynamic_obstacles

# ---- Configuration ----
dt = 1.0
config = load_yaml("/app/commonocean-sim/src/configuration.yaml")
config["general_simulator"]["dt"] = dt
config["general_simulator"]["using_collision_avoider"] = False
config["general_simulator"]["using_collision_detection"] = True
config["general_simulator"]["using_displayer"] = True
config["general_simulator"]["maximum_runtime"] = 2000
config["general_simulator"]["plotting"]["do_plotting"] = False

# ---- Load scenario XML ----
scenario_path = "/app/commonocean-sim/tutorials/scenarios/ZAM_AAA-2_20250129_T-1.xml"
if len(sys.argv) > 1:
    scenario_path = sys.argv[1]

print(f"Loading scenario: {scenario_path}")
reader = CommonOceanFileReader(scenario_path)
scenario, planning_problem_set = reader.open()

# Interpolate traffic trajectories to match sim dt (e.g. 10s → 1s)
interpolate_dynamic_obstacles(scenario, dt)

# ---- Create COLAV-controlled vessel from FIRST planning problem only ----
# Other planning problems remain as-is in the scenario
colav_params = dict(a=1.67, v=12.0, eta=3.5, tp=3.0, Cs=300.0)

models = []
for idx, pp in enumerate(planning_problem_set._planning_problem_dict.values()):
    if idx > 0:
        print(f"  Planning problem {pp.planning_problem_id}: "
              f"Leaving as scenario planning problem (not simulator-controlled)")
        continue

    # Extract waypoints from planning problem
    # Start with initial position — divide_long_waypointpaths needs at least 2 points.
    init = pp.initial_state
    waypoints = [np.array(init.position, dtype="float64")]
    if pp.waypoints:
        waypoints.extend(pp.generate_reference_points_from_waypoint())
    goal_pos = extract_center_point_of_region(pp.goal)
    waypoints.append(goal_pos)
    # Overshoot waypoint so the vessel sails through the goal rectangle
    # instead of stopping ~87.5m short (final_waypoint_finished_radius).
    approach_dir = goal_pos - waypoints[-2]
    approach_dist = np.linalg.norm(approach_dir)
    if approach_dist > 1e-6:
        overshoot = goal_pos + (approach_dir / approach_dist) * 200.0
    else:
        overshoot = goal_pos + np.array([200.0, 0.0])
    waypoints.append(overshoot)
    waypoints = np.array(waypoints, dtype="float64")
    init_state = YPState(
        position=np.array(init.position, dtype="float64"),
        velocity=init.velocity,
        orientation=init.orientation,
        time_step=0,
    )

    # Use the scenario's actual velocity
    scenario_params = {**colav_params, 'v': init.velocity}
    colav_factory = ColavVesselFactory(dt, config, **scenario_params)

    vessel = colav_factory.get_vessel(
        init_state, waypoints, dt,
        ship_name=f"COLAV_{pp.planning_problem_id}",
        vesselid=pp.planning_problem_id,
        goal_waypoint=(goal_pos[0], goal_pos[1]),
    )
    models.append(vessel)
    print(f"  Vessel {pp.planning_problem_id} [COLAV CONTROLLER]: "
          f"pos={init.position}, heading={np.degrees(init.orientation):.1f}deg, "
          f"v={init.velocity} m/s, {len(waypoints)} waypoints")

# ---- Dynamic obstacles (traffic vessels from scenario) ----
dynamic_obstacles = list(scenario.dynamic_obstacles)
print(f"  {len(dynamic_obstacles)} dynamic obstacle(s)")

# ---- Static obstacles ----
static_obstacles = list(scenario.static_obstacles)

# ---- Build and run simulation ----
simfac = SimulatorFactory(dt)
simfac.configure_simulation_factory(
    models=models,
    dynObsts=dynamic_obstacles,
    obsts=static_obstacles,
    runnerlist=[],
    states_rate=1,
    current_configuration=config,
)

sim = simfac.generate_scenario()

# generate_scenario() deep-copies models, so fix up the copies
for m in sim.models:
    if hasattr(m, 'controller') and hasattr(m.controller, 'sim'):
        m.controller.sim = sim
        m.controller.controlled_vessel = m

# Shutdown original (unused) controller's asyncio thread
for v in models:
    if hasattr(v, 'controller') and hasattr(v.controller, 'shutdown'):
        v.controller.shutdown()

print("Running simulation...")
sim.display_run()

# Print diagnostics
for m in sim.models:
    ctrl = getattr(m, 'controller', None)
    if ctrl and hasattr(ctrl, 'stepped'):
        print(f"  {m.vessel_name}: {ctrl.stepped} steps, "
              f"final pos={m.position}")

print("Done.")
