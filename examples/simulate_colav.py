#!/usr/bin/env python3
"""
COLAV Automaton Simulation Script

Uses hybrid_automaton_runner framework with rich visualization.
Supports multiple scenarios and provides comprehensive trajectory analysis.

Usage:
    python simulate_colav.py                    # Run all predefined scenarios
    python simulate_colav.py --scenario 1       # Run specific scenario (1, 2, or 3)
    python simulate_colav.py --interactive      # Interactive mode for custom scenarios
"""

import sys
from pathlib import Path

# Script directory for relative imports and output
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR.parent / 'src'))

import argparse
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Any, Optional

from colav_automaton import ColavAutomaton
from hybrid_automaton import Automaton
from hybrid_automaton_runner import AutomatonRunner
from colav_automaton.unsafe_sets import get_unsafe_set_vertices
from colav_automaton.integration import normalize_heading_in_results


# =============================================================================
# PREDEFINED SCENARIOS
# =============================================================================

SCENARIOS = {
    1: {
        "name": "Single Stationary Obstacle",
        "description": "Basic avoidance of a single stationary obstacle on the path",
        "waypoint": (20.0, 20.0),
        "obstacles": [(10.0, 10.0, 0.0, 0.0)],  # (x, y, velocity, heading)
        "initial_state": [0.0, 0.0, np.pi/4],  # x, y, psi (heading toward waypoint)
        "duration": 10.0,
        "Cs": 3.0,
        "tp": 1.0,
    },
    2: {
        "name": "Multiple Obstacles - Crowded Environment",
        "description": "Navigation through multiple stationary obstacles",
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
    },
    3: {
        "name": "Head-On Encounter",
        "description": "Slow-moving obstacle approaching head-on",
        "waypoint": (100.0, 0.0),
        "obstacles": [
            (70.0, 0.0, 0.5, np.pi),      # Moving toward ship at 0.5 m/s (very slow)
        ],
        "initial_state": [0.0, 0.0, 0.0],  # Heading east
        "duration": 15.0,
        "Cs": 5.0,
        "tp": 1.0,
    },
    4: {
        "name": "Crossing Encounter",
        "description": "Slow-moving obstacle crossing from starboard",
        "waypoint": (60.0, 0.0),
        "obstacles": [
            (30.0, -12.0, 2.0, np.pi/2),  # Moving north at 2 m/s, closer to path
        ],
        "initial_state": [0.0, 0.0, 0.0],  # Heading east
        "duration": 12.0,
        "Cs": 8.0,   # Large safety radius
        "tp": 2.5,
    },
}


# =============================================================================
# SIMULATION CLASS
# =============================================================================

class ColavSimulation:
    """
    Runs COLAV automaton simulation using AutomatonRunner
    and provides visualization capabilities.
    """

    def __init__(
        self,
        waypoint: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float]],
        initial_state: List[float],
        duration: float = 10.0,
        Cs: float = 2.0,
        v: float = 12.0,
        a: float = 1.67,
        eta: float = 3.5,
        tp: float = 1.0,
        v1_buffer: float = 0.0,
        dt: float = 0.01,
        sampling_rate: float = 0.001,
    ):
        """
        Initialize simulation.

        Args:
            waypoint: Target (x, y) position
            obstacles: List of (ox, oy, velocity, heading) tuples
            initial_state: Initial [x, y, psi] state
            duration: Simulation duration in seconds
            Cs: Safety radius around obstacles
            v: Ship velocity (m/s)
            a: System dynamics parameter
            eta: Controller gain
            tp: Prescribed time for controller convergence
            v1_buffer: Buffer distance (m) to offset V1 to starboard for extra clearance
            dt: Integration timestep
            sampling_rate: Runner sampling rate
        """
        self.waypoint = waypoint
        self.obstacles = obstacles
        self.initial_state = np.array(initial_state, dtype=float)
        self.duration = duration
        self.Cs = Cs
        self.v = v
        self.a = a
        self.eta = eta
        self.tp = tp
        self.v1_buffer = v1_buffer
        self.dt = dt
        self.sampling_rate = sampling_rate

        # Computed parameters
        self.dsafe = Cs + (v * 2) * tp

        # Results storage
        self.results = None
        self.automaton = None
        self.runner = None

    def create_automaton(self) -> Automaton:
        """Create the COLAV automaton with current parameters."""
        self.automaton = ColavAutomaton(
            waypoint_x=self.waypoint[0],
            waypoint_y=self.waypoint[1],
            obstacles=self.obstacles,
            Cs=self.Cs,
            a=self.a,
            v=self.v,
            eta=self.eta,
            tp=self.tp,
            v1_buffer=self.v1_buffer,
        )
        return self.automaton

    async def run(self, verbose: bool = True, waypoint_threshold: float = 0.2) -> Dict[str, Any]:
        """
        Run the simulation using AutomatonRunner.

        Args:
            verbose: Print progress information
            waypoint_threshold: Distance to waypoint to consider "reached" (default 2.0m)

        Returns:
            Dictionary containing simulation results
        """
        if verbose:
            print(f"  Creating automaton...")

        self.create_automaton()

        if verbose:
            print(f"  Initializing runner (dt={self.dt}, sampling={self.sampling_rate})...")

        self.runner = AutomatonRunner(self.automaton, sampling_rate=self.sampling_rate)

        if verbose:
            print(f"  Running simulation for up to {self.duration}s...")

        # Run with waypoint checking
        await self._run_with_waypoint_check(waypoint_threshold, verbose)

        self.results = self.runner.get_results()

        # Post-process to normalize headings
        self.results = normalize_heading_in_results(self.results)

        if verbose:
            self._print_summary()

        return self.results

    async def _run_with_waypoint_check(self, threshold: float, verbose: bool):
        """Run simulation and stop when waypoint is reached or duration exceeded."""
        import asyncio

        # Start the runner tasks manually
        self.runner.clear_all_data()
        self.runner._tasks = []

        # Main automaton task
        ha_task = asyncio.create_task(
            self.automaton.activate(
                x0=self.initial_state,
                aux_x0={},
                real_time_mode=False,
                integrate=True,
                dt=self.dt
            )
        )
        self.runner._tasks.append(ha_task)

        # Collector tasks
        self.runner._tasks.append(
            asyncio.create_task(self.runner.continuous_collector.collect(self.automaton))
        )
        self.runner._tasks.append(
            asyncio.create_task(self.runner.automaton_collector.collect(self.automaton))
        )
        self.runner._tasks.append(
            asyncio.create_task(self.runner.transition_collector.collect(self.automaton))
        )

        # Watchdog that checks both duration and waypoint distance
        while True:
            await asyncio.sleep(0.0001)
            try:
                sim_time = self.automaton.get_runtime_time_elapsed()
                state = self.automaton.get_runtime_continuous_state().latest()

                # Check waypoint reached
                dist = np.sqrt(
                    (state[0] - self.waypoint[0])**2 +
                    (state[1] - self.waypoint[1])**2
                )

                if dist < threshold:
                    if verbose:
                        print(f"  Waypoint reached at t={sim_time:.2f}s (dist={dist:.2f}m)")
                    break

                if sim_time >= self.duration:
                    break

            except Exception:
                pass

        # Cancel all tasks
        for task in self.runner._tasks:
            task.cancel()
        await asyncio.gather(*self.runner._tasks, return_exceptions=True)

    def _print_summary(self):
        """Print simulation summary statistics."""
        if self.results is None:
            return

        continuous_states = self.results.get('continuous_states', [])
        automaton_states = self.results.get('automaton_states', [])
        transition_times = self.results.get('transition_times', [])

        if not continuous_states:
            print("  No results collected!")
            return

        # Final position
        final_time, final_state = continuous_states[-1]
        final_x, final_y = final_state[0], final_state[1]
        dist_to_waypoint = np.sqrt(
            (final_x - self.waypoint[0])**2 +
            (final_y - self.waypoint[1])**2
        )

        # State time analysis
        state_times = {'WAYPOINT_REACHING': 0.0, 'COLLISION_AVOIDANCE': 0.0, 'CONSTANT_CONTROL': 0.0}
        if automaton_states:
            prev_time = 0.0
            prev_state = automaton_states[0][1] if automaton_states else 'WAYPOINT_REACHING'
            for t, state_name in automaton_states:
                if prev_state in state_times:
                    state_times[prev_state] += t - prev_time
                prev_time = t
                prev_state = state_name
            # Add remaining time
            if prev_state in state_times:
                state_times[prev_state] += final_time - prev_time

        print(f"\n  Summary:")
        print(f"    Total time: {final_time:.2f}s")
        print(f"    Final position: ({final_x:.2f}, {final_y:.2f})")
        print(f"    Distance to waypoint: {dist_to_waypoint:.2f}m")
        print(f"    Transitions: {len(transition_times)}")
        print(f"    Time in S1 (Waypoint): {state_times['WAYPOINT_REACHING']:.2f}s")
        print(f"    Time in S2 (Avoidance): {state_times['COLLISION_AVOIDANCE']:.2f}s")
        print(f"    Time in S3 (Constant): {state_times['CONSTANT_CONTROL']:.2f}s")


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class SimulationVisualizer:
    """Visualize COLAV simulation results."""

    STATE_COLORS = {
        'WAYPOINT_REACHING': 'blue',
        'COLLISION_AVOIDANCE': 'red',
        'CONSTANT_CONTROL': 'orange',
    }

    def __init__(self, simulation: ColavSimulation):
        """
        Initialize visualizer.

        Args:
            simulation: Completed ColavSimulation instance
        """
        self.sim = simulation
        self.results = simulation.results

    def _get_virtual_waypoints(self) -> List[Tuple[float, float]]:
        """Get virtual waypoints from the collision avoidance controller."""
        if self.sim.automaton is None:
            return []
        cfg = self.sim.automaton._definition.get_configuration()
        if 'ca_controller' not in cfg:
            return []
        return cfg['ca_controller'].virtual_waypoint_history

    def plot_trajectory(
        self,
        title: str = "COLAV Trajectory",
        show_unsafe_sets: bool = True,
        show_virtual_waypoints: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create trajectory visualization.

        Args:
            title: Plot title
            show_unsafe_sets: Whether to show unsafe set polygons
            figsize: Figure size
            save_path: If provided, save figure to this path

        Returns:
            matplotlib Figure
        """
        if self.results is None:
            raise ValueError("No simulation results available")

        continuous_states = self.results.get('continuous_states', [])
        automaton_states = self.results.get('automaton_states', [])

        if not continuous_states:
            raise ValueError("No continuous states recorded")

        # Extract trajectory data
        times = np.array([s[0] for s in continuous_states])
        states = np.array([s[1] for s in continuous_states])
        traj_x = states[:, 0]
        traj_y = states[:, 1]

        # Map times to automaton states
        state_at_time = self._get_state_at_times(times, automaton_states)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot trajectory with state-based coloring
        self._plot_colored_trajectory(ax, traj_x, traj_y, state_at_time)

        # Plot start and end positions
        ax.plot(traj_x[0], traj_y[0], 'go', markersize=15,
                label='Start', zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(traj_x[-1], traj_y[-1], 'bs', markersize=12,
                label='End', zorder=10, markeredgecolor='darkblue', markeredgewidth=2)

        # Plot waypoint
        ax.plot(self.sim.waypoint[0], self.sim.waypoint[1], 'r*', markersize=25,
                label='Waypoint', zorder=10, markeredgecolor='darkred', markeredgewidth=1)

        # Plot obstacles
        self._plot_obstacles(ax)

        # Plot unsafe sets at key trajectory points
        if show_unsafe_sets:
            self._plot_unsafe_sets(ax, traj_x, traj_y)

        # Plot virtual waypoints
        if show_virtual_waypoints:
            virtual_wps = self._get_virtual_waypoints()
            if virtual_wps:
                vwp_x = [vw[0] for vw in virtual_wps]
                vwp_y = [vw[1] for vw in virtual_wps]
                ax.scatter(vwp_x, vwp_y, c='magenta', marker='D', s=80,
                          label='Virtual Waypoints (V1)', zorder=9,
                          edgecolors='darkmagenta', linewidths=1.5)

        # Create legend
        self._add_legend(ax)

        # Formatting
        self._format_plot(ax, traj_x, traj_y, title)

        # Add summary text box
        self._add_summary_box(ax, times[-1], traj_x[-1], traj_y[-1], state_at_time)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        return fig

    def plot_states_over_time(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create multi-panel plot showing states over time.

        Args:
            figsize: Figure size
            save_path: If provided, save figure to this path

        Returns:
            matplotlib Figure
        """
        continuous_states = self.results.get('continuous_states', [])
        automaton_states = self.results.get('automaton_states', [])

        times = np.array([s[0] for s in continuous_states])
        states = np.array([s[1] for s in continuous_states])

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # XY trajectory with time coloring
        ax1 = axes[0, 0]
        self._plot_xy_with_time_color(ax1, states[:, 0], states[:, 1], times)
        ax1.set_title('XY Position (colored by time)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # X and Y over time
        ax2 = axes[0, 1]
        ax2.plot(times, states[:, 0], 'b-', label='X', linewidth=2)
        ax2.plot(times, states[:, 1], 'r-', label='Y', linewidth=2)
        ax2.axhline(y=self.sim.waypoint[0], color='b', linestyle='--', alpha=0.5, label=f'X target={self.sim.waypoint[0]}')
        ax2.axhline(y=self.sim.waypoint[1], color='r', linestyle='--', alpha=0.5, label=f'Y target={self.sim.waypoint[1]}')
        ax2.set_title('Position over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Heading over time
        ax3 = axes[1, 0]
        headings_deg = np.rad2deg(states[:, 2])
        ax3.plot(times, headings_deg, 'g-', linewidth=2)
        ax3.set_title('Heading over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heading (deg)')
        ax3.grid(True, alpha=0.3)

        # Automaton state over time
        ax4 = axes[1, 1]
        self._plot_automaton_states(ax4, automaton_states, times[-1])
        ax4.set_title('Automaton State')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('State')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        return fig

    def _get_state_at_times(self, times: np.ndarray, automaton_states: List) -> List[str]:
        """Map timestamps to automaton state names."""
        if not automaton_states:
            return ['WAYPOINT_REACHING'] * len(times)

        state_at_time = []
        state_idx = 0
        current_state = automaton_states[0][1] if automaton_states else 'WAYPOINT_REACHING'

        for t in times:
            while state_idx < len(automaton_states) and automaton_states[state_idx][0] <= t:
                current_state = automaton_states[state_idx][1]
                state_idx += 1
            state_at_time.append(current_state)

        return state_at_time

    def _plot_colored_trajectory(self, ax, traj_x, traj_y, state_at_time):
        """Plot trajectory with state-based coloring."""
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Map states to numeric values
        state_to_num = {'WAYPOINT_REACHING': 0, 'COLLISION_AVOIDANCE': 1, 'CONSTANT_CONTROL': 2}
        state_nums = np.array([state_to_num.get(s, 0) for s in state_at_time])
        segment_colors = (state_nums[:-1] + state_nums[1:]) / 2

        colors = ['blue', 'red', 'orange']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], len(colors))

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, zorder=5)
        lc.set_array(segment_colors)
        ax.add_collection(lc)

    def _plot_xy_with_time_color(self, ax, x, y, times):
        """Plot XY trajectory colored by time."""
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=times.min(), vmax=times.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
        lc.set_array(times[:-1])

        line = ax.add_collection(lc)
        ax.figure.colorbar(line, ax=ax, label='Time (s)')

        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'rs', markersize=10, label='End')
        ax.legend()

    def _plot_automaton_states(self, ax, automaton_states, max_time):
        """Plot automaton state as step function."""
        if not automaton_states:
            return

        state_to_num = {'WAYPOINT_REACHING': 1, 'COLLISION_AVOIDANCE': 2, 'CONSTANT_CONTROL': 3}

        times = [0.0] + [s[0] for s in automaton_states] + [max_time]
        state_names = [automaton_states[0][1]] + [s[1] for s in automaton_states]
        state_nums = [state_to_num.get(s, 1) for s in state_names]
        state_nums.append(state_nums[-1])

        ax.step(times, state_nums, where='post', linewidth=2, color='purple')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['S1: Waypoint', 'S2: Avoidance', 'S3: Constant'])
        ax.set_ylim(0.5, 3.5)

    def _plot_obstacles(self, ax):
        """Plot obstacles with safety circles and velocity arrows."""
        for i, (ox, oy, ov, o_psi) in enumerate(self.sim.obstacles):
            # Safety circle
            circle = Circle((ox, oy), self.sim.Cs, color='orange', alpha=0.3,
                           edgecolor='darkorange', linewidth=2, zorder=3)
            ax.add_patch(circle)

            # Obstacle center
            ax.plot(ox, oy, 'o', color='darkorange', markersize=8, zorder=4)

            # Velocity arrow (if moving)
            if ov > 0:
                arrow_len = max(self.sim.Cs * 2, ov * 2)
                dx = arrow_len * np.cos(o_psi)
                dy = arrow_len * np.sin(o_psi)
                ax.arrow(ox, oy, dx, dy, head_width=0.5, head_length=0.3,
                        fc='darkorange', ec='darkorange', alpha=0.7, zorder=4)

            # Label
            ax.text(ox, oy - self.sim.Cs - 1.0, f'O{i+1}', fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    def _plot_unsafe_sets(self, ax, traj_x, traj_y):
        """Plot unsafe set polygons at key trajectory points."""
        n_points = len(traj_x)
        key_indices = [0, n_points // 3, 2 * n_points // 3, n_points - 1]

        for idx in key_indices:
            x, y = traj_x[idx], traj_y[idx]
            unsafe_verts = get_unsafe_set_vertices(
                x, y, self.sim.obstacles, self.sim.Cs, dsf=self.sim.dsafe
            )
            if unsafe_verts and len(unsafe_verts) >= 3:
                alpha = 0.08 if idx != n_points - 1 else 0.12
                poly = MplPolygon(unsafe_verts, color='red', alpha=alpha,
                                 edgecolor='red', linewidth=1, linestyle='--', zorder=1)
                ax.add_patch(poly)

    def _add_legend(self, ax):
        """Add custom legend with state colors and markers."""
        legend_elements = [
            Patch(facecolor='blue', label='S1: Waypoint Reaching'),
            Patch(facecolor='red', label='S2: Collision Avoidance'),
            Patch(facecolor='orange', label='S3: Constant Control'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='magenta',
                   markeredgecolor='darkmagenta', markersize=10, label='Virtual Waypoint (V1)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)

    def _format_plot(self, ax, traj_x, traj_y, title):
        """Apply plot formatting."""
        # Compute bounds
        all_x = list(traj_x) + [self.sim.waypoint[0]] + [o[0] for o in self.sim.obstacles]
        all_y = list(traj_y) + [self.sim.waypoint[1]] + [o[1] for o in self.sim.obstacles]

        x_min, x_max = min(all_x) - 3, max(all_x) + 3
        y_min, y_max = min(all_y) - 3, max(all_y) + 3

        # Ensure minimum range
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 15:
            x_center = (x_min + x_max) / 2
            x_min, x_max = x_center - 7.5, x_center + 7.5
        if y_range < 15:
            y_center = (y_min + y_max) / 2
            y_min, y_max = y_center - 7.5, y_center + 7.5

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')

    def _add_summary_box(self, ax, total_time, final_x, final_y, state_at_time):
        """Add summary statistics text box."""
        dist_to_wp = np.sqrt(
            (final_x - self.sim.waypoint[0])**2 +
            (final_y - self.sim.waypoint[1])**2
        )

        # Count time in each state
        state_counts = {'WAYPOINT_REACHING': 0, 'COLLISION_AVOIDANCE': 0, 'CONSTANT_CONTROL': 0}
        for s in state_at_time:
            if s in state_counts:
                state_counts[s] += 1

        # Convert counts to approximate time (assuming uniform sampling)
        dt_approx = total_time / len(state_at_time) if state_at_time else 0

        status = f"""Navigation Summary:
Total Time: {total_time:.1f}s
Distance to WP: {dist_to_wp:.2f}m
Obstacles: {len(self.sim.obstacles)}
S2 Time: ~{state_counts['COLLISION_AVOIDANCE'] * dt_approx:.1f}s"""

        ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
               fontweight='bold')


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

async def run_scenario(
    scenario_id: int,
    show_plots: bool = True,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run a predefined scenario.

    Args:
        scenario_id: Scenario number (1, 2, or 3)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files

    Returns:
        Simulation results dictionary
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_id}. Available: {list(SCENARIOS.keys())}")

    scenario = SCENARIOS[scenario_id]

    print(f"\n{'='*70}")
    print(f"Scenario {scenario_id}: {scenario['name']}")
    print(f"{'='*70}")
    print(f"  {scenario['description']}")
    print(f"  Waypoint: {scenario['waypoint']}")
    print(f"  Obstacles: {len(scenario['obstacles'])}")

    # Create and run simulation
    sim = ColavSimulation(
        waypoint=scenario['waypoint'],
        obstacles=scenario['obstacles'],
        initial_state=scenario['initial_state'],
        duration=scenario['duration'],
        Cs=scenario['Cs'],
        tp=scenario['tp'],
        v1_buffer=scenario.get('v1_buffer', 3.0),
    )

    results = await sim.run(verbose=True)

    # Visualize
    if results and results.get('continuous_states'):
        viz = SimulationVisualizer(sim)

        # Trajectory plot
        fig1 = viz.plot_trajectory(
            title=f"Scenario {scenario_id}: {scenario['name']}",
            save_path=str(OUTPUT_DIR / f"scenario{scenario_id}_trajectory.png") if save_plots else None,
        )

        # State analysis plot
        fig2 = viz.plot_states_over_time(
            save_path=str(OUTPUT_DIR / f"scenario{scenario_id}_analysis.png") if save_plots else None,
        )

        if show_plots:
            plt.show()
        else:
            plt.close('all')

    return results


async def run_custom_scenario(
    waypoint: Tuple[float, float],
    obstacles: List[Tuple[float, float, float, float]],
    initial_state: Optional[List[float]] = None,
    duration: float = 10.0,
    v1_buffer: float = 0.0,
    name: str = "Custom Scenario",
    show_plots: bool = True,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run a custom scenario with user-defined parameters.

    Args:
        waypoint: Target (x, y) position
        obstacles: List of (ox, oy, velocity, heading) tuples
        initial_state: Initial [x, y, psi] (default: origin heading toward waypoint)
        duration: Simulation duration
        v1_buffer: Buffer distance (m) to offset V1 to starboard for extra clearance
        name: Scenario name for plot titles
        show_plots: Whether to display plots
        save_plots: Whether to save plots

    Returns:
        Simulation results dictionary
    """
    # Default initial state: origin, heading toward waypoint
    if initial_state is None:
        angle_to_wp = np.arctan2(waypoint[1], waypoint[0])
        initial_state = [0.0, 0.0, angle_to_wp]

    print(f"\n{'='*70}")
    print(f"Custom: {name}")
    print(f"{'='*70}")
    print(f"  Waypoint: {waypoint}")
    print(f"  Obstacles: {len(obstacles)}")

    sim = ColavSimulation(
        waypoint=waypoint,
        obstacles=obstacles,
        initial_state=initial_state,
        duration=duration,
        v1_buffer=v1_buffer,
    )

    results = await sim.run(verbose=True)

    if results and results.get('continuous_states'):
        viz = SimulationVisualizer(sim)

        fig1 = viz.plot_trajectory(
            title=name,
            save_path=str(OUTPUT_DIR / "custom_trajectory.png") if save_plots else None,
        )

        fig2 = viz.plot_states_over_time(
            save_path=str(OUTPUT_DIR / "custom_analysis.png") if save_plots else None,
        )

        if show_plots:
            plt.show()
        else:
            plt.close('all')

    return results


async def run_all_scenarios(show_plots: bool = True, save_plots: bool = True):
    """Run all predefined scenarios."""
    print("\n" + "="*70)
    print("COLAV AUTOMATON SIMULATION - All Scenarios")
    print("="*70)

    results = {}
    for scenario_id in SCENARIOS:
        results[scenario_id] = await run_scenario(
            scenario_id,
            show_plots=False,  # Don't show intermediate
            save_plots=save_plots,
        )

    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETE")
    print("="*70)

    if save_plots:
        print(f"\nGenerated files in {OUTPUT_DIR}:")
        for sid in SCENARIOS:
            print(f"  - scenario{sid}_trajectory.png")
            print(f"  - scenario{sid}_analysis.png")

    if show_plots:
        plt.show()

    return results


def interactive_mode():
    """Interactive mode for custom scenario definition."""
    print("\n" + "="*70)
    print("COLAV AUTOMATON - Interactive Mode")
    print("="*70)

    print("\nDefine your scenario:")

    # Waypoint
    wp_input = input("  Waypoint (x,y) [default: 20,20]: ").strip()
    if wp_input:
        wx, wy = map(float, wp_input.split(','))
        waypoint = (wx, wy)
    else:
        waypoint = (20.0, 20.0)

    # Obstacles
    print("  Enter obstacles as: x,y,velocity,heading (one per line, empty to finish)")
    obstacles = []
    while True:
        obs_input = input(f"    Obstacle {len(obstacles)+1}: ").strip()
        if not obs_input:
            break
        parts = list(map(float, obs_input.split(',')))
        if len(parts) == 2:
            parts.extend([0.0, 0.0])  # Default stationary
        elif len(parts) == 3:
            parts.append(0.0)  # Default heading
        obstacles.append(tuple(parts[:4]))

    if not obstacles:
        obstacles = [(10.0, 10.0, 0.0, 0.0)]  # Default single obstacle
        print("    Using default obstacle: (10, 10, 0, 0)")

    # Duration
    dur_input = input("  Duration (seconds) [default: 10]: ").strip()
    duration = float(dur_input) if dur_input else 10.0

    return waypoint, obstacles, duration


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COLAV Automaton Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate_colav.py                    # Run all scenarios
  python simulate_colav.py --scenario 1       # Run scenario 1 only
  python simulate_colav.py --interactive      # Custom scenario
  python simulate_colav.py --no-show          # Generate plots without displaying
        """
    )
    parser.add_argument('--scenario', '-s', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='Run specific scenario (1, 2, or 3)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode for custom scenarios')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to files')

    args = parser.parse_args()

    show_plots = not args.no_show
    save_plots = not args.no_save

    try:
        if args.interactive:
            waypoint, obstacles, duration = interactive_mode()
            asyncio.run(run_custom_scenario(
                waypoint=waypoint,
                obstacles=obstacles,
                duration=duration,
                show_plots=show_plots,
                save_plots=save_plots,
            ))
        elif args.scenario:
            asyncio.run(run_scenario(
                args.scenario,
                show_plots=show_plots,
                save_plots=save_plots,
            ))
        else:
            asyncio.run(run_all_scenarios(
                show_plots=show_plots,
                save_plots=save_plots,
            ))
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
