"""
Run a commonocean-sim XML scenario with the COLAV automaton controller.

Usage inside the Docker container:
    cd /app/commonocean-sim/src
    python3 /app/usv-navigation/examples/commonocean_scenario.py
"""

import sys
import numpy as np
from rules.common.helper import load_yaml
from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.scenario.state import YPState
from Simulator.SimulatorFactory import SimulatorFactory
from Pipeline.SimulationIO import extract_center_point_of_region

from colav_automaton.adapters import ColavVesselFactory

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

# ---- Create COLAV-controlled vessel from FIRST planning problem only ----
# Other planning problems remain as-is in the scenario
colav_factory = ColavVesselFactory(
    dt, config,
    a=1.67, v=12.0, eta=3.5, tp=1.0, Cs=300.0,
)

models = []
for idx, pp in enumerate(planning_problem_set._planning_problem_dict.values()):
    if idx > 0:
        # Only control the first vessel with COLAV
        print(f"  Planning problem {pp.planning_problem_id}: "
              f"Leaving as scenario planning problem (not simulator-controlled)")
        continue

    # Extract waypoints from planning problem
    if pp.waypoints:
        waypoints = pp.generate_reference_points_from_waypoint()
    else:
        waypoints = []

    goal_pos = extract_center_point_of_region(pp.goal)
    waypoints.append(goal_pos)
    waypoints = np.array(waypoints, dtype="float64")

    # Build initial state (YP model)
    init = pp.initial_state
    init_state = YPState(
        position=np.array(init.position, dtype="float64"),
        velocity=init.velocity,
        orientation=init.orientation,
        time_step=0,
    )

    vessel = colav_factory.get_vessel(
        init_state, waypoints, dt,
        ship_name=f"COLAV_{pp.planning_problem_id}",
        vesselid=pp.planning_problem_id,
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

# Give adapters access to the simulator (for obstacle queries)
for m in sim.models:
    if hasattr(m, 'controller') and hasattr(m.controller, 'sim'):
        m.controller.sim = sim

print("Running simulation...")
sim.display_run()

# Print diagnostics
for m in sim.models:
    ctrl = getattr(m, 'controller', None)
    if ctrl and hasattr(ctrl, 'stepped'):
        print(f"  {m.vessel_name}: {ctrl.stepped} steps, "
              f"final pos={m.position}")

print("Done.")
