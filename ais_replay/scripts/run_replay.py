#!/usr/bin/env python3
"""
Run the COLAV automaton through a recorded AIS scenario and render the
result as a GIF + trajectory PNG.

No Docker or simulator required — only the core package and matplotlib.

Usage (from the repo root):
    # Bundled synthetic sample (Singapore Strait geometry)
    PYTHONPATH=src:. python3 ais_replay/scripts/run_replay.py

    # A real recording captured with record_ais.py
    PYTHONPATH=src:. python3 ais_replay/scripts/run_replay.py \
        --recording my_harbor.jsonl \
        --ego-start 1.195,103.84 --goal 1.205,103.89

Coordinates are lat,lon (WGS84); the local frame origin is the mean of
the recording's positions unless --origin is given.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

from ais_replay.geo import LocalFrame
from ais_replay.runner import ReplayRunner
from ais_replay.sources import RecordedAISSource
from ais_replay.tracker import TrafficTracker

DEFAULT_RECORDING = SCRIPT_DIR.parent / "sample_data" / "sample_strait.jsonl"
# Defaults matched to the bundled sample's geometry (west -> east transit)
DEFAULT_EGO_START = "1.2000,103.8500"
DEFAULT_GOAL = "1.2000,103.8880"

STATE_COLORS = {
    "WAYPOINT_REACHING": "blue",
    "COLLISION_AVOIDANCE": "red",
    "CONSTANT_CONTROL": "orange",
}


def parse_latlon(text):
    lat, lon = (float(p) for p in text.split(","))
    return lat, lon


def render(runner, out_gif, out_png, fps=15, max_frames=300):
    """Render trajectory PNG (always) and GIF (when out_gif is not None)."""
    pos = np.array(runner.position_tracker)
    n = len(pos)
    goal = runner.goal

    all_x = [p[0] for p in pos] + [goal[0]]
    all_y = [p[1] for p in pos] + [goal[1]]
    for frame_obs in runner.traffic_tracker:
        for ox, oy, _, _ in frame_obs:
            all_x.append(ox)
            all_y.append(oy)
    pad = 400.0
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    # --- static trajectory plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for state, color in STATE_COLORS.items():
        mask = [s == state for s in runner.state_tracker]
        ax.plot(pos[mask, 0], pos[mask, 1], ".", ms=2, color=color,
                label=state.replace("_", " ").title())
    for frame_obs in runner.traffic_tracker[::10]:
        for ox, oy, _, _ in frame_obs:
            ax.plot(ox, oy, "s", ms=1, color="darkred", alpha=0.3)
    ax.plot(*goal, "g*", ms=18, label="Goal")
    ax.plot(pos[0, 0], pos[0, 1], "b^", ms=10, label="Start")
    ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.set_aspect("equal"), ax.grid(alpha=0.3), ax.legend(loc="upper right")
    ax.set_xlabel("x east (m)"), ax.set_ylabel("y north (m)")
    ax.set_title("AIS replay trajectory")
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    if out_gif is None:
        return

    # --- animation ---
    step = max(1, n // max_frames)
    indices = list(range(0, n, step))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.set_aspect("equal"), ax.grid(alpha=0.3)
    ax.plot(*goal, "g*", ms=18, zorder=5)
    traj_line, = ax.plot([], [], "b-", lw=1.2, zorder=10)
    ego_dot, = ax.plot([], [], "bo", ms=9, zorder=15)
    traffic_dots, = ax.plot([], [], "rs", ms=7, zorder=12)
    v1_dot, = ax.plot([], [], "m^", ms=10, zorder=14)
    unsafe_patch = MplPolygon([[0, 0]], closed=True, fc="purple",
                              alpha=0.18, ec="purple", zorder=8)
    ax.add_patch(unsafe_patch)
    state_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         fontsize=11, fontweight="bold", va="top",
                         bbox=dict(fc="white", alpha=0.8))

    def update(i):
        idx = indices[i]
        x, y, _ = pos[idx]
        traj_line.set_data(pos[:idx + 1, 0], pos[:idx + 1, 1])
        ego_dot.set_data([x], [y])
        obs = runner.traffic_tracker[idx]
        traffic_dots.set_data([o[0] for o in obs], [o[1] for o in obs])
        v1 = runner.v1_tracker[idx]
        v1_dot.set_data([v1[0]] if v1 else [], [v1[1]] if v1 else [])
        coords = runner.unsafe_set_tracker[idx]
        unsafe_patch.set_xy(coords if coords else [[xlim[0] - 1e6, 0]])
        state = runner.state_tracker[idx]
        state_text.set_text(f"{state}\nt = {runner.times[idx]:.0f} s")
        state_text.set_color(STATE_COLORS.get(state, "black"))
        return traj_line, ego_dot, traffic_dots, v1_dot, unsafe_patch, state_text

    anim = FuncAnimation(fig, update, frames=len(indices), blit=True)
    anim.save(out_gif, writer="pillow", fps=fps, dpi=72)
    plt.close(fig)
    print(f"Wrote {out_gif}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=str(DEFAULT_RECORDING))
    ap.add_argument("--ego-start", default=DEFAULT_EGO_START,
                    help="ego start lat,lon")
    ap.add_argument("--goal", default=DEFAULT_GOAL, help="goal lat,lon")
    ap.add_argument("--origin", default=None,
                    help="local frame origin lat,lon (default: ego start)")
    ap.add_argument("--ego-v", type=float, default=6.0, help="ego speed m/s")
    ap.add_argument("--Cs", type=float, default=300.0)
    ap.add_argument("--tp", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--pace", type=float, default=0.02,
                    help="wall seconds to sleep per tick (default keeps "
                         "the validated wall:sim ratio; dt = real time)")
    ap.add_argument("--max-duration", type=float, default=3600.0)
    ap.add_argument("--obstacle-range", type=float, default=8000.0,
                    help="only pass tracks within this range of the ego "
                         "to the automaton (m); 0 = no filter")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "ais_replay"))
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    ego_lat, ego_lon = parse_latlon(args.ego_start)
    goal_lat, goal_lon = parse_latlon(args.goal)
    origin = parse_latlon(args.origin) if args.origin else (ego_lat, ego_lon)

    frame = LocalFrame(*origin)
    source = RecordedAISSource(args.recording)
    tracker = TrafficTracker(frame)

    ego_xy = frame.to_xy(ego_lat, ego_lon)
    goal_xy = frame.to_xy(goal_lat, goal_lon)
    ego_psi = float(np.arctan2(goal_xy[1] - ego_xy[1], goal_xy[0] - ego_xy[0]))

    print(f"Recording: {args.recording} "
          f"({len(source.reports)} reports, "
          f"{source.t_end - source.t_start:.0f} s span)")
    print(f"Ego: {args.ego_start} -> {args.goal} at {args.ego_v} m/s")

    runner = ReplayRunner(
        source, tracker,
        ego_start=(ego_xy[0], ego_xy[1], ego_psi),
        goal=goal_xy,
        v=args.ego_v, Cs=args.Cs, tp=args.tp,
        dt=args.dt, pace=args.pace, max_duration=args.max_duration,
        obstacle_range=args.obstacle_range,
    )
    summary = runner.run()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = Path(args.recording).stem
    out_png = os.path.join(args.output_dir, f"{stem}_trajectory.png")
    out_gif = None if args.no_gif else os.path.join(args.output_dir, f"{stem}.gif")
    render(runner, out_gif, out_png)

    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
