#!/usr/bin/env python3
"""
Real-time COLAV Simulation Visualization

Animates the simulation showing:
- Agent vessel motion with heading indicator
- Dynamic obstacle motion
- LOS cone from vessel to waypoint
- Unsafe regions around obstacles
- Virtual waypoint V1 when in avoidance mode
- State indicator (S1/S2/S3)
"""

import sys
import asyncio
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'src'))

import os
import matplotlib
# Use WebAgg if no display, otherwise try TkAgg
if not os.environ.get('DISPLAY'):
    matplotlib.use('WebAgg')
    print("No display found. Animation will open in web browser at http://127.0.0.1:8988")
else:
    matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon, FancyArrow
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional

from colav_automaton import ColavAutomaton
from hybrid_automaton import Automaton
from hybrid_automaton_runner import AutomatonRunner
from colav_automaton.unsafe_sets import get_unsafe_set_vertices, create_los_cone


# Scenario configurations (matching simulate_colav.py)
SCENARIOS = {
    1: {
        "name": "Single Stationary Obstacle",
        "waypoint": (20.0, 20.0),
        "obstacles": [(10.0, 10.0, 0.0, 0.0)],
        "initial_state": [0.0, 0.0, np.pi/4],
        "duration": 10.0,
        "Cs": 3.0,
        "tp": 1.0,
        "v1_buffer": 3.0,
    },
    2: {
        "name": "Multiple Obstacles - Crowded Environment",
        "waypoint": (28.0, 28.0),
        "obstacles": [
            (7.0, 7.0, 0.0, 0.0),
            (13.0, 13.0, 0.0, 0.0),
            (9.0, 15.0, 0.0, 0.0),
            (15.0, 9.0, 0.0, 0.0),
            (11.0, 11.0, 0.0, 0.0),
        ],
        "initial_state": [0.0, 0.0, np.pi/4],
        "duration": 12.0,
        "Cs": 2.0,
        "tp": 1.0,
        "v1_buffer": 3.0,
    },
    3: {
        "name": "Head-On Encounter",
        "waypoint": (100.0, 0.0),
        "obstacles": [(70.0, 0.0, 2.0, np.pi)],
        "initial_state": [0.0, 0.0, 0.0],
        "duration": 15.0,
        "Cs": 5.0,
        "tp": 1.0,
        "v1_buffer": 3.0,
    },
    4: {
        "name": "Crossing Encounter",
        "waypoint": (60.0, 0.0),
        "obstacles": [(30.0, -12.0, 2.0, np.pi/2)],
        "initial_state": [0.0, 0.0, 0.0],
        "duration": 12.0,
        "Cs": 8.0,
        "tp": 2.5,
        "v1_buffer": 3.0,
    },
    5: {
        "name": "Overtaking Encounter",
        "description": "Ship overtaking a slower vessel moving in the same direction",
        "waypoint": (100.0, 0.0),
        "obstacles": [(40.0, 0.0, 4.0, 0.0)],  # Moving east at 4 m/s (slower than ship's 12 m/s)
        "initial_state": [0.0, 0.0, 0.0],
        "duration": 15.0,
        "Cs": 5.0,
        "tp": 1.5,
        "v1_buffer": 3.0,
    },
    6: {
        "name": "Multi-Vessel Crossing",
        "description": "Two dynamic obstacles crossing from different directions",
        "waypoint": (100.0, 0.0),
        "obstacles": [
            (30.0, -10.0, 2.0, np.pi/2),    # Crossing from south, moving north
            (60.0, 10.0, 2.0, -np.pi/2),    # Crossing from north, moving south
        ],
        "initial_state": [0.0, 0.0, 0.0],
        "duration": 18.0,
        "Cs": 5.0,
        "tp": 1.5,
        "v1_buffer": 5.0,
    },
}


class RealtimeSimulation:
    """Real-time animated simulation visualization."""

    def __init__(
        self,
        scenario_id: int = 1,
        speed_multiplier: float = 1.0,
        show_los_cone: bool = True,
        show_unsafe_sets: bool = True,
    ):
        self.scenario_id = scenario_id
        self.scenario = SCENARIOS[scenario_id]
        self.speed_multiplier = speed_multiplier
        self.show_los_cone = show_los_cone
        self.show_unsafe_sets = show_unsafe_sets

        # Simulation parameters
        self.waypoint = self.scenario["waypoint"]
        self.initial_obstacles = self.scenario["obstacles"]
        self.initial_state = np.array(self.scenario["initial_state"])
        self.duration = self.scenario["duration"]
        self.Cs = self.scenario["Cs"]
        self.tp = self.scenario["tp"]
        self.v1_buffer = self.scenario.get("v1_buffer", 3.0)
        self.v = 12.0
        self.dsafe = self.Cs + (self.v * 2) * self.tp

        # Pre-computed simulation data
        self.sim_times = []
        self.sim_states = []
        self.sim_modes = []
        self.sim_v1 = []
        self.frame_idx = 0
        self.automaton = None
        self.runner = None

        # Animation objects
        self.fig = None
        self.ax = None
        self.vessel_marker = None
        self.heading_arrow = None
        self.trajectory_line = None
        self.los_cone_patch = None
        self.obstacle_markers = []
        self.unsafe_set_patches = []
        self.v1_marker = None
        self.state_text = None
        self.time_text = None

    def setup_automaton(self):
        """Create and initialize the automaton."""
        self.automaton = ColavAutomaton(
            waypoint_x=self.waypoint[0],
            waypoint_y=self.waypoint[1],
            obstacles=self.initial_obstacles,
            Cs=self.Cs,
            tp=self.tp,
            v=self.v,
            v1_buffer=self.v1_buffer,
        )
        self.runner = AutomatonRunner(self.automaton, sampling_rate=0.01)

    def get_current_obstacles(self, t: float) -> List[Tuple[float, float, float, float]]:
        """Get obstacle positions at time t (accounting for motion)."""
        obstacles = []
        for ox, oy, ov, o_psi in self.initial_obstacles:
            # Update position based on velocity and heading
            new_x = ox + ov * np.cos(o_psi) * t
            new_y = oy + ov * np.sin(o_psi) * t
            obstacles.append((new_x, new_y, ov, o_psi))
        return obstacles

    def setup_plot(self):
        """Initialize the plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # Determine plot bounds
        all_x = [self.initial_state[0], self.waypoint[0]]
        all_y = [self.initial_state[1], self.waypoint[1]]
        for ox, oy, ov, o_psi in self.initial_obstacles:
            all_x.extend([ox, ox + ov * self.duration * np.cos(o_psi)])
            all_y.extend([oy, oy + ov * self.duration * np.sin(o_psi)])

        margin = max(self.dsafe, 15)
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f"COLAV Simulation: {self.scenario['name']}")

        # Plot waypoint
        self.ax.plot(self.waypoint[0], self.waypoint[1], 'r*', markersize=20,
                     label='Waypoint', zorder=20)

        # Initialize trajectory line
        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2,
                                              alpha=0.7, label='Trajectory')

        # Initialize vessel marker
        self.vessel_marker, = self.ax.plot([], [], 'bo', markersize=12, zorder=15)

        # Initialize heading arrow (will be updated)
        self.heading_arrow = None

        # Initialize obstacle markers
        for i, (ox, oy, ov, o_psi) in enumerate(self.initial_obstacles):
            marker, = self.ax.plot(ox, oy, 'rs', markersize=10, zorder=10)
            self.obstacle_markers.append(marker)

            # Draw obstacle safety circle
            circle = Circle((ox, oy), self.Cs, color='orange', alpha=0.3,
                           fill=True, zorder=5)
            self.ax.add_patch(circle)

        # Initialize LOS cone patch
        self.los_cone_patch = None

        # Initialize V1 marker
        self.v1_marker, = self.ax.plot([], [], 'mD', markersize=12,
                                        label='Virtual Waypoint V1', zorder=12)

        # State and time text
        self.state_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=14, verticalalignment='top',
                                        fontweight='bold',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.time_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes,
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='Agent Vessel'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=10, label='Obstacle'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=15, label='Waypoint'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='magenta',
                   markersize=10, label='Virtual Waypoint V1'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

    def update_los_cone(self, pos_x, pos_y, psi):
        """Update the LOS cone visualization."""
        if self.los_cone_patch is not None:
            self.los_cone_patch.remove()
            self.los_cone_patch = None

        if not self.show_los_cone:
            return

        # Create LOS cone polygon
        try:
            effective_tp = min(self.tp, 1.0)
            los_cone = create_los_cone(pos_x, pos_y,
                                       self.waypoint[0], self.waypoint[1],
                                       self.v, effective_tp)
            if los_cone is not None and los_cone.is_valid:
                x, y = los_cone.exterior.xy
                self.los_cone_patch = MplPolygon(
                    list(zip(x, y)),
                    color='cyan', alpha=0.2, zorder=3,
                    label='LOS Cone'
                )
                self.ax.add_patch(self.los_cone_patch)
        except Exception:
            pass

    def update_unsafe_sets(self, pos_x, pos_y, psi, obstacles):
        """Update unsafe set visualizations."""
        # Remove old patches
        for patch in self.unsafe_set_patches:
            patch.remove()
        self.unsafe_set_patches = []

        if not self.show_unsafe_sets:
            return

        try:
            vertices = get_unsafe_set_vertices(
                pos_x, pos_y, obstacles, self.Cs,
                dsf=self.dsafe, ship_psi=psi, ship_v=self.v,
                use_swept_region=False
            )
            if vertices and len(vertices) >= 3:
                patch = MplPolygon(
                    vertices,
                    facecolor='red', alpha=0.15, zorder=4,
                    edgecolor='darkred', linewidth=1
                )
                self.ax.add_patch(patch)
                self.unsafe_set_patches.append(patch)
        except Exception:
            pass

    def update_heading_arrow(self, x, y, psi):
        """Update the heading direction arrow."""
        if self.heading_arrow is not None:
            self.heading_arrow.remove()

        arrow_length = 3.0
        dx = arrow_length * np.cos(psi)
        dy = arrow_length * np.sin(psi)

        self.heading_arrow = FancyArrow(
            x, y, dx, dy,
            width=0.8, head_width=1.5, head_length=0.8,
            fc='darkblue', ec='black', zorder=16
        )
        self.ax.add_patch(self.heading_arrow)

    def run_simulation(self):
        """Run the full simulation and store results."""
        print("Running simulation...")

        results = asyncio.run(self.runner.run(
            x0=self.initial_state,
            duration=self.duration,
            dt=0.01,  # Match simulate_colav.py
            collect_continuous=True,
            collect_automaton=True,
        ))

        # Extract trajectory data - format is [[time, state], ...]
        continuous = results.get('continuous_states', [])
        automaton = results.get('automaton_states', [])

        # Extract times and states from continuous data
        self.sim_times = [entry[0] for entry in continuous]
        self.sim_states = [entry[1] for entry in continuous]

        # Extract modes from automaton data
        mode_map = {
            'WAYPOINT_REACHING': 'S1',
            'COLLISION_AVOIDANCE': 'S2',
            'CONSTANT_CONTROL': 'S3',
        }
        self.sim_modes = []
        for entry in automaton:
            mode_name = entry[1] if len(entry) > 1 else 'WAYPOINT_REACHING'
            self.sim_modes.append(mode_map.get(mode_name, 'S1'))

        # Pad modes to match trajectory length if needed
        while len(self.sim_modes) < len(self.sim_states):
            self.sim_modes.append(self.sim_modes[-1] if self.sim_modes else 'S1')

        # Truncate simulation when waypoint is reached (like simulate_colav.py)
        waypoint_threshold = 0.5
        for i, state in enumerate(self.sim_states):
            dist = np.sqrt((state[0] - self.waypoint[0])**2 + (state[1] - self.waypoint[1])**2)
            if dist < waypoint_threshold:
                self.sim_times = self.sim_times[:i+1]
                self.sim_states = self.sim_states[:i+1]
                self.sim_modes = self.sim_modes[:i+1]
                print(f"Waypoint reached at t={self.sim_times[-1]:.2f}s")
                break

        # Store virtual waypoint history with timestamps
        # Find when each S2 period started to associate V1s with times
        self.virtual_waypoints = []
        self.v1_times = []  # Time when each V1 was created
        try:
            cfg = self.automaton._definition.get_configuration()
            if 'ca_controller' in cfg and cfg['ca_controller'] is not None:
                self.virtual_waypoints = list(cfg['ca_controller'].virtual_waypoint_history)

                # Find S2 entry times to associate with V1s
                s2_entries = []
                for i in range(1, len(self.sim_modes)):
                    if self.sim_modes[i] == 'S2' and self.sim_modes[i-1] != 'S2':
                        s2_entries.append(self.sim_times[i])

                # Each V1 corresponds to an S2 entry
                for i, v1 in enumerate(self.virtual_waypoints):
                    if i < len(s2_entries):
                        self.v1_times.append(s2_entries[i])
                    else:
                        self.v1_times.append(0)  # Fallback
        except Exception:
            pass

        print(f"Simulation complete: {len(self.sim_states)} frames, {len(self.virtual_waypoints)} virtual waypoints")

    def animate(self, frame):
        """Animation update function."""
        if frame >= len(self.sim_states):
            return []

        state = self.sim_states[frame]
        t = self.sim_times[frame] if frame < len(self.sim_times) else 0
        mode = self.sim_modes[frame] if frame < len(self.sim_modes) else 'S1'

        x, y, psi = state[0], state[1], state[2]
        obstacles = self.get_current_obstacles(t)

        # Update trajectory (show path up to current frame)
        traj_x = [s[0] for s in self.sim_states[:frame+1]]
        traj_y = [s[1] for s in self.sim_states[:frame+1]]
        self.trajectory_line.set_data(traj_x, traj_y)

        # Update vessel position
        self.vessel_marker.set_data([x], [y])

        # Update heading arrow
        self.update_heading_arrow(x, y, psi)

        # Update obstacle positions
        for i, (ox, oy, ov, o_psi) in enumerate(obstacles):
            if i < len(self.obstacle_markers):
                self.obstacle_markers[i].set_data([ox], [oy])

        # Update LOS cone
        self.update_los_cone(x, y, psi)

        # Update unsafe sets
        self.update_unsafe_sets(x, y, psi, obstacles)

        # Update V1 marker - only show V1s that have been created by current time
        if self.virtual_waypoints and self.v1_times:
            v1_x = []
            v1_y = []
            for i, (vw, v1_time) in enumerate(zip(self.virtual_waypoints, self.v1_times)):
                if t >= v1_time:  # Only show V1 after it was created
                    v1_x.append(vw[0])
                    v1_y.append(vw[1])
            self.v1_marker.set_data(v1_x, v1_y)
        else:
            self.v1_marker.set_data([], [])

        # Update state text with color coding
        state_colors = {'S1': 'blue', 'S2': 'red', 'S3': 'orange'}
        state_names = {
            'S1': 'S1: WAYPOINT_REACHING',
            'S2': 'S2: COLLISION_AVOIDANCE',
            'S3': 'S3: CONSTANT_CONTROL'
        }
        mode_key = mode[:2]
        self.state_text.set_text(state_names.get(mode_key, mode))
        self.state_text.set_color(state_colors.get(mode_key, 'black'))

        # Update time text
        self.time_text.set_text(f't = {t:.2f}s')

        # Check if reached waypoint
        dist_to_wp = np.sqrt((x - self.waypoint[0])**2 + (y - self.waypoint[1])**2)
        if dist_to_wp < 0.5:
            self.time_text.set_text(f't = {t:.2f}s (ARRIVED)')

        return [self.trajectory_line, self.vessel_marker, self.v1_marker,
                self.state_text, self.time_text]

    def run(self):
        """Run the real-time simulation."""
        print(f"\nStarting real-time simulation: {self.scenario['name']}")
        print(f"Speed multiplier: {self.speed_multiplier}x")
        print("Close the window to stop.\n")

        # Setup automaton and run simulation first
        self.setup_automaton()
        self.run_simulation()

        # Setup plot after simulation
        self.setup_plot()

        # Calculate animation parameters
        # Frame interval in ms (base 50fps, adjusted by speed)
        base_interval = 20  # 20ms = 50fps
        interval = int(base_interval / self.speed_multiplier)
        interval = max(10, min(interval, 200))  # Clamp between 10-200ms

        num_frames = len(self.sim_states)

        self.anim = FuncAnimation(
            self.fig, self.animate,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=True
        )

        plt.show()

    def save_animation(self, output_path: Optional[str] = None):
        """Save the animation as a GIF file."""
        print(f"\nSaving animation for: {self.scenario['name']}")

        # Setup automaton and run simulation
        self.setup_automaton()
        self.run_simulation()

        # Use Agg backend for saving
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Setup plot
        self.setup_plot()

        # Calculate animation parameters
        interval = 50  # 50ms per frame for GIF

        num_frames = len(self.sim_states)

        self.anim = FuncAnimation(
            self.fig, self.animate,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=False
        )

        # Determine output path
        if output_path is None:
            scenario_name = self.scenario['name'].lower().replace(' ', '_').replace('-', '_')
            output_path = OUTPUT_DIR / f"scenario{self.scenario_id}_{scenario_name}.gif"

        # Save as GIF
        print(f"Saving to: {output_path}")
        self.anim.save(str(output_path), writer='pillow', fps=20)
        print(f"Saved: {output_path}")
        plt.close(self.fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time COLAV Simulation")
    parser.add_argument('--scenario', '-s', type=int, default=1,
                        choices=list(range(1, 11)),
                        help='Scenario number (1-6)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed multiplier (default: 1.0)')
    parser.add_argument('--no-los', action='store_true',
                        help='Hide LOS cone')
    parser.add_argument('--no-unsafe', action='store_true',
                        help='Hide unsafe sets')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as GIF to output directory')
    parser.add_argument('--save-all', action='store_true',
                        help='Save animations for all scenarios')

    args = parser.parse_args()

    if args.save_all:
        # Save all scenarios
        print(f"Saving all scenarios to {OUTPUT_DIR}/")
        for scenario_id in SCENARIOS:
            sim = RealtimeSimulation(
                scenario_id=scenario_id,
                show_los_cone=not args.no_los,
                show_unsafe_sets=not args.no_unsafe,
            )
            sim.save_animation()
        print(f"\nAll animations saved to {OUTPUT_DIR}/")
    elif args.save:
        # Save single scenario
        sim = RealtimeSimulation(
            scenario_id=args.scenario,
            show_los_cone=not args.no_los,
            show_unsafe_sets=not args.no_unsafe,
        )
        sim.save_animation()
    else:
        # Run interactive animation
        sim = RealtimeSimulation(
            scenario_id=args.scenario,
            speed_multiplier=args.speed,
            show_los_cone=not args.no_los,
            show_unsafe_sets=not args.no_unsafe,
        )
        sim.run()


if __name__ == "__main__":
    main()
