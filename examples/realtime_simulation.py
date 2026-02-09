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
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'src'))

import os
import matplotlib
# Use Agg backend for saving (non-interactive)
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon, FancyArrow
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional

from colav_automaton import ColavAutomaton
from hybrid_automaton import Automaton, RunResult
from colav_automaton.controllers import get_unsafe_set_vertices, create_los_cone, HeadingControlProvider


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

        # Animation objects
        self.fig = None
        self.ax = None
        self.vessel_marker = None
        self.heading_arrow = None
        self.trajectory_line = None
        self.los_cone_patch = None
        self.obstacle_markers = []
        self.unsafe_set_patches = []
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
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

    def update_los_cone(self, pos_x, pos_y, psi):
        """Update the LOS cone visualization."""
        # TODO: LOS cone temporarily disabled - visualization doesn't match guard logic
        return

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

        run_output_dir = str(OUTPUT_DIR / f"run_scenario{self.scenario_id}")

        controller = HeadingControlProvider(self.automaton)

        results: RunResult = asyncio.run(self.automaton.activate(
            initial_continuous_state=np.array(self.initial_state, dtype=float),
            initial_control_input_states={'u': np.array([0.0])},
            enable_real_time_mode=False,
            enable_self_integration=True,
            delta_time=0.01,
            timeout_sec=self.duration,
            continuous_state_sampler_enabled=True,
            continuous_state_sampler_rate=100,
            control_states_provider=controller,
            control_states_provision_rate=100,
            output_dir=run_output_dir,
        ))

        # Read continuous state trajectory from sampler CSV
        csv_path = os.path.join(run_output_dir, "continuous_state.csv")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                t = float(row[0])
                state = json.loads(row[1])
                self.sim_times.append(t)
                self.sim_states.append(state)

        # Parse mode transitions from temporal log
        log_path = os.path.join(run_output_dir, "temporal_automaton.log")
        transitions = []
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(
                    r"\[INFO\]\s*\[(\d+\.?\d*)\].*\[Transition\]\s*'(\w+)'\s*--\[.*\]-->\s*'(\w+)'",
                    line
                )
                if match:
                    t = float(match.group(1))
                    to_state = match.group(3)
                    transitions.append((t, to_state))

        # Build mode list for each timestep
        mode_map = {
            'WAYPOINT_REACHING': 'S1',
            'COLLISION_AVOIDANCE': 'S2',
            'CONSTANT_CONTROL': 'S3',
        }
        current_mode = 'WAYPOINT_REACHING'
        trans_idx = 0
        for t in self.sim_times:
            while trans_idx < len(transitions) and transitions[trans_idx][0] <= t:
                current_mode = transitions[trans_idx][1]
                trans_idx += 1
            self.sim_modes.append(mode_map.get(current_mode, 'S1'))

        # Truncate simulation when waypoint is reached
        waypoint_threshold = 0.5
        for i, state in enumerate(self.sim_states):
            dist = np.sqrt((state[0] - self.waypoint[0])**2 + (state[1] - self.waypoint[1])**2)
            if dist < waypoint_threshold:
                self.sim_times = self.sim_times[:i+1]
                self.sim_states = self.sim_states[:i+1]
                self.sim_modes = self.sim_modes[:i+1]
                print(f"Waypoint reached at t={self.sim_times[-1]:.2f}s")
                break

        print(f"Simulation complete: {len(self.sim_states)} samples")

        # Subsample for animation: target 30fps real-time playback
        self.anim_fps = 30
        sim_duration = self.sim_times[-1] - self.sim_times[0] if len(self.sim_times) > 1 else 1.0
        target_frames = max(2, int(sim_duration * self.anim_fps))
        n = len(self.sim_states)
        if n > target_frames:
            step = n / target_frames
            indices = [int(i * step) for i in range(target_frames)]
            if indices[-1] != n - 1:
                indices.append(n - 1)
            self.anim_indices = indices
        else:
            self.anim_indices = list(range(n))

        print(f"Animation: {len(self.anim_indices)} frames at {self.anim_fps}fps ({sim_duration:.1f}s real-time)")

    def animate(self, frame):
        """Animation update function."""
        if frame >= len(self.anim_indices):
            return []

        idx = self.anim_indices[frame]
        state = self.sim_states[idx]
        t = self.sim_times[idx] if idx < len(self.sim_times) else 0
        mode = self.sim_modes[idx] if idx < len(self.sim_modes) else 'S1'

        x, y, psi = state[0], state[1], state[2]
        obstacles = self.get_current_obstacles(t)

        # Update trajectory (show full path up to current simulation index)
        traj_x = [s[0] for s in self.sim_states[:idx+1]]
        traj_y = [s[1] for s in self.sim_states[:idx+1]]
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

        return [self.trajectory_line, self.vessel_marker,
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

        # Frame interval for real-time playback, adjusted by speed multiplier
        interval = int(1000 / (self.anim_fps * self.speed_multiplier))
        interval = max(10, min(interval, 200))

        self.anim = FuncAnimation(
            self.fig, self.animate,
            frames=len(self.anim_indices),
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

        # Setup plot
        self.setup_plot()

        # Animation at real-time speed
        interval = int(1000 / self.anim_fps)

        self.anim = FuncAnimation(
            self.fig, self.animate,
            frames=len(self.anim_indices),
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
        self.anim.save(str(output_path), writer='pillow', fps=self.anim_fps)
        print(f"Saved: {output_path}")
        plt.close(self.fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time COLAV Simulation - Saves animations as GIF files")
    parser.add_argument('--scenario', '-s', type=int, default=1,
                        choices=list(range(1, 11)),
                        help='Scenario number (1-6)')
    parser.add_argument('--no-los', action='store_true',
                        help='Hide LOS cone')
    parser.add_argument('--no-unsafe', action='store_true',
                        help='Hide unsafe sets')
    parser.add_argument('--all', action='store_true',
                        help='Save animations for all scenarios')

    args = parser.parse_args()

    if args.all:
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
    else:
        # Save single scenario (default behavior)
        sim = RealtimeSimulation(
            scenario_id=args.scenario,
            show_los_cone=not args.no_los,
            show_unsafe_sets=not args.no_unsafe,
        )
        sim.save_animation()


if __name__ == "__main__":
    main()
