"""
Collision test: two vessels on a collision course.

Runs a COLAV-controlled vessel (East-bound) against an MPC vessel (West-bound)
and generates trajectory plots + an animated GIF of the encounter.

Usage inside the Docker container:
    cd /app/commonocean-sim/src
    python3 /app/usv-navigation/examples/commonocean_collision_test.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from rules.common.helper import load_yaml
from commonocean.scenario.state import YPState
from Simulator.SimulatorFactory import SimulatorFactory
from Environment.VesselFactory import SControlledYawParamVesselFactory

from colav_automaton.adapters import ColavVesselFactory
from colav_automaton.controllers.unsafe_sets import get_unsafe_set_vertices

# ---- Configuration ----
dt = 1.0
config = load_yaml("/app/commonocean-sim/src/configuration.yaml")
config["general_simulator"]["dt"] = dt
config["general_simulator"]["using_collision_avoider"] = False
config["general_simulator"]["using_collision_detection"] = True
config["general_simulator"]["using_displayer"] = True
config["general_simulator"]["maximum_runtime"] = 600
config["general_simulator"]["plotting"]["do_plotting"] = False

OUTPUT_DIR = "/app/usv-navigation/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Creating collision test scenario...")

# ---- Vessel 1: COLAV controller (East-bound) ----
colav_factory = ColavVesselFactory(
    dt, config,
    a=1.67, v=5.0, eta=3.5, tp=3.0, Cs=400.0,
    v1_buffer=90.0,
)

init_state_1 = YPState(
    position=np.array([0.0, 0.0], dtype="float64"),
    velocity=5.0,
    orientation=0.0,  # East
    time_step=0,
)
waypoints_1 = np.array([
    [800.0, 0.0],
    [1600.0, 0.0],
    [2400.0, 0.0],
], dtype="float64")

vessel_1 = colav_factory.get_vessel(
    init_state_1, waypoints_1, dt,
    ship_name="COLAV_East", vesselid=10,
)

# ---- Vessel 2: Default MPC (West-bound) ----
mpc_factory = SControlledYawParamVesselFactory(dt, config)

init_state_2 = YPState(
    position=np.array([2400.0, 50.0], dtype="float64"),
    velocity=5.0,
    orientation=np.pi,  # West
    time_step=0,
)
waypoints_2 = np.array([
    [1600.0, 50.0],
    [800.0, 50.0],
    [0.0, 50.0],
    [-800.0, 50.0],
], dtype="float64")

vessel_2 = mpc_factory.get_vessel(
    init_state_2, waypoints_2, dt,
    ship_name="MPC_West", vesselid=20,
)

print(f"  COLAV: pos=(0,0) heading=0deg v=5m/s")
print(f"  MPC:   pos=(2400,50) heading=180deg v=5m/s")
print(f"  Separation: 2400m, closing speed: 10m/s")

# ---- Build and run simulation ----
simfac = SimulatorFactory(dt)
simfac.configure_simulation_factory(
    models=[vessel_1, vessel_2],
    dynObsts=[], obsts=[], runnerlist=[],
    states_rate=1,
    current_configuration=config,
)

sim = simfac.generate_scenario()

for m in sim.models:
    if hasattr(m, 'controller') and hasattr(m.controller, 'sim'):
        m.controller.sim = sim

print("Running collision test...")
sim.display_run()

# ---- Diagnostics ----
colav_vessel = None
for m in sim.models:
    ctrl = getattr(m, 'controller', None)
    if ctrl and hasattr(ctrl, 'stepped'):
        print(f"  {m.vessel_name}: {ctrl.stepped} steps, final pos={m.position}")
        if 'COLAV' in m.vessel_name:
            colav_vessel = m

if colav_vessel is None:
    print("Error: Could not find COLAV vessel in simulation")
    exit(1)

# ---- Extract trajectory data ----
colav_ctrl = colav_vessel.controller
colav_positions = np.array(colav_ctrl.position_tracker)
colav_states = colav_ctrl.state_tracker
colav_v1_waypoints = colav_ctrl.v1_tracker

if len(colav_positions) == 0:
    print("Warning: No positions recorded, cannot generate plot")
    exit(0)

# Approximate MPC trajectory (straight west from start)
mpc_positions = []
pos2 = np.array([2400.0, 50.0])
for i in range(len(colav_positions)):
    mpc_positions.append(pos2.copy())
    pos2 = pos2 + 5.0 * dt * np.array([np.cos(np.pi), np.sin(np.pi)])
    if pos2[0] < -800:
        break
mpc_positions = np.array(mpc_positions)

Cs_val = colav_ctrl.Cs
v1_buffer_val = colav_ctrl.v1_buffer
safe_distance = Cs_val + v1_buffer_val

# ---- Static trajectory plot ----
print("Generating trajectory plot...")
fig, ax = plt.subplots(figsize=(14, 10))

ax.plot(colav_positions[:, 0], colav_positions[:, 1], 'b-', linewidth=2, label='COLAV Vessel', zorder=3)
ax.plot(0, 0, 'bo', markersize=10, label='COLAV Start')
ax.plot(colav_positions[-1, 0], colav_positions[-1, 1], 'b^', markersize=10, label='COLAV End')

ax.plot(mpc_positions[:, 0], mpc_positions[:, 1], 'r-', linewidth=2, label='MPC Vessel', zorder=3)
ax.plot(2400, 50, 'ro', markersize=10, label='MPC Start')

for wp in waypoints_1:
    ax.plot(wp[0], wp[1], 'g*', markersize=15, zorder=4)
ax.plot(waypoints_1[-1, 0], waypoints_1[-1, 1], 'g*', markersize=20, label='COLAV Goal', zorder=4)

# Plot unsafe sets at ~10 sample points during approach
min_len = min(len(colav_positions), len(mpc_positions))
unsafe_set_plotted = False
for i in range(0, min_len, max(1, min_len // 10)):
    colav_pos = colav_positions[i]
    mpc_pos = mpc_positions[i]
    obstacles = [(mpc_pos[0], mpc_pos[1], 5.0, np.pi)]

    unsafe_vertices = get_unsafe_set_vertices(
        ship_x=colav_pos[0], ship_y=colav_pos[1],
        obstacles_list=obstacles, Cs=Cs_val,
        ship_psi=colav_pos[2], ship_v=5.0,
        use_swept_region=True,
    )

    if unsafe_vertices and len(unsafe_vertices) >= 3:
        poly = np.array(unsafe_vertices)
        poly = np.vstack([poly, poly[0]])
        label = 'Dynamic Unsafe Set' if not unsafe_set_plotted else None
        ax.plot(poly[:, 0], poly[:, 1], 'orange', linewidth=1.5, alpha=0.6, label=label, zorder=1)
        ax.fill(poly[:, 0], poly[:, 1], 'orange', alpha=0.15, zorder=1)
        unsafe_set_plotted = True

# Mark closest approach
closest_idx, min_dist = 0, float('inf')
for i in range(min_len):
    dist = np.linalg.norm(colav_positions[i, :2] - mpc_positions[i])
    if dist < min_dist:
        min_dist = dist
        closest_idx = i

if closest_idx < len(mpc_positions):
    circle = mpatches.Circle(mpc_positions[closest_idx], safe_distance,
                             fill=False, edgecolor='red', linewidth=2,
                             linestyle='--', label=f'Collision Zone (Cs={Cs_val}m)', zorder=2)
    ax.add_patch(circle)
    ax.plot(mpc_positions[closest_idx, 0], mpc_positions[closest_idx, 1],
            'rx', markersize=15, label=f'Closest Approach ({min_dist:.1f}m)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Collision Avoidance Test\nCs={Cs_val}m, v1_buffer={v1_buffer_val}m, safe_distance={safe_distance}m')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'collision_test_trajectory.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.close()

# ---- Animated GIF ----
MODE_LABELS = {
    'WAYPOINT_REACHING': ('S1: WAYPOINT_REACHING', 'blue'),
    'COLLISION_AVOIDANCE': ('S2: COLLISION_AVOIDANCE', 'red'),
    'CONSTANT_CONTROL': ('S3: CONSTANT_CONTROL', 'orange'),
}
FRAME_SKIP = 5
num_frames = min_len // FRAME_SKIP


def animate(frame_idx):
    ax.clear()
    t = frame_idx * FRAME_SKIP

    current_state = colav_states[t] if t < len(colav_states) else 'WAYPOINT_REACHING'
    state_label, state_color = MODE_LABELS.get(current_state, (current_state, 'black'))

    # Trajectory history
    if t > 0:
        ax.plot(colav_positions[:t, 0], colav_positions[:t, 1], 'b-', linewidth=2, alpha=0.5)
        ax.plot(mpc_positions[:t, 0], mpc_positions[:t, 1], 'r-', linewidth=2, alpha=0.5)

    if t < len(colav_positions) and t < len(mpc_positions):
        colav_pos = colav_positions[t]
        mpc_pos = mpc_positions[t]
        colav_heading = colav_pos[2]

        # Unsafe set
        obstacles = [(mpc_pos[0], mpc_pos[1], 5.0, np.pi)]
        unsafe_vertices = get_unsafe_set_vertices(
            ship_x=colav_pos[0], ship_y=colav_pos[1],
            obstacles_list=obstacles, Cs=Cs_val,
            ship_psi=colav_heading, ship_v=5.0,
            use_swept_region=True,
        )
        if unsafe_vertices and len(unsafe_vertices) >= 3:
            poly = np.array(unsafe_vertices)
            poly = np.vstack([poly, poly[0]])
            ax.fill(poly[:, 0], poly[:, 1], 'orange', alpha=0.3, zorder=2)
            ax.plot(poly[:, 0], poly[:, 1], 'orange', linewidth=2, alpha=0.8, zorder=2)

        # Collision zone
        ax.add_patch(mpatches.Circle(mpc_pos, Cs_val, fill=False, edgecolor='red',
                                     linewidth=2, linestyle='--', alpha=0.5, zorder=2))

        # Vessel positions + heading arrows
        ax.plot(colav_pos[0], colav_pos[1], 'bo', markersize=12, label='COLAV Vessel', zorder=5)
        ax.plot(mpc_pos[0], mpc_pos[1], 'rs', markersize=12, label='MPC Vessel', zorder=5)
        arrow_len = 50
        ax.arrow(colav_pos[0], colav_pos[1],
                 arrow_len * np.cos(colav_heading), arrow_len * np.sin(colav_heading),
                 head_width=20, head_length=20, fc='blue', ec='blue', linewidth=2, zorder=5)
        ax.arrow(mpc_pos[0], mpc_pos[1],
                 arrow_len * np.cos(np.pi), arrow_len * np.sin(np.pi),
                 head_width=20, head_length=20, fc='red', ec='red', linewidth=2, zorder=5)

    # Waypoints
    for wp in waypoints_1:
        ax.plot(wp[0], wp[1], 'g*', markersize=15, zorder=4)
    ax.plot(waypoints_1[-1, 0], waypoints_1[-1, 1], 'g*', markersize=20, label='Goal', zorder=4)

    # Virtual waypoint V1 (during collision avoidance)
    if t < len(colav_v1_waypoints) and colav_v1_waypoints[t] is not None:
        v1 = colav_v1_waypoints[t]
        ax.plot(v1[0], v1[1], 'c*', markersize=18,
                markeredgecolor='black', markeredgewidth=1.5,
                label='Virtual Waypoint (V1)', zorder=6)

    # Labels
    ax.text(0.02, 0.98, state_label, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', fontweight='bold', color=state_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.02, 0.90, f't = {t * dt:.0f}s', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('COLAV Simulation: Collision Avoidance Test')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 2600)
    ax.set_ylim(-600, 300)
    ax.set_aspect('equal')


print("Generating animated GIF...")
fig, ax = plt.subplots(figsize=(14, 10))
anim = FuncAnimation(fig, animate, frames=num_frames, interval=200, repeat=True)
gif_path = os.path.join(OUTPUT_DIR, 'collision_test_animation.gif')
anim.save(gif_path, writer=PillowWriter(fps=5), dpi=100)
print(f"  Saved: {gif_path}")
plt.close()

print("Done.")
