# CommonOcean integration

The adapter and evaluation harness that drive the COLAV automaton inside
[commonocean-sim](https://github.com/CommonOcean/commonocean-sim). Everything
simulator-specific lives in this folder — the core automaton in
[`src/colav_automaton/`](../src/colav_automaton/) has no CommonOcean dependencies.

## Contents

| Path | Purpose |
|---|---|
| `adapters/controller.py` | `HybridAutomatonController` — steps `SyncColavRuntime.step_external` each simulator tick |
| `adapters/vessel_factory.py` | `ColavVesselFactory` — creates the automaton-controlled vessel |
| `evaluation/metrics.py` | Per-scenario metrics: CPA, goal reached, collision, encounter type, state times |
| `sim_utils.py` | Shared utilities: trajectory interpolation, config loading |
| `scripts/` | Scenario runners, batch evaluators, visualization tools |

## Docker stack

The full simulation stack (commonocean-sim + Gurobi + VNC) runs in Docker:

```bash
# Interactive shell
docker/start.sh -it

# Or detached (access via VNC at http://localhost:6080/vnc.html)
docker/start.sh
```

The compose file builds the two algorithm dependencies from sibling checkouts
(`../hybrid-automaton`, `../unsafe-set`) and mounts `../commonocean-scenarios/scenarios`
as `/app/scenarios`, so keep those repositories next to this one;
`commonocean-sim` itself is cloned inside the image (see
[`docker/docker-compose.yml`](../docker/docker-compose.yml)). This folder and `src/`
are volume-mounted, so source edits apply without rebuilding.

Inside the container, scripts run from `/app/commonocean-sim/src` with
`PYTHONPATH=/app/commonocean-sim/src:/app/usv-navigation` (preconfigured).

## Single-scenario runs

**Head-on collision test** (COLAV vessel East-bound vs MPC vessel West-bound):

```bash
cd /app/commonocean-sim/src
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_collision_test.py
```

Saves a trajectory plot and animated GIF to `/app/usv-navigation/output/`.

**A single CommonOcean XML scenario:**

```bash
cd /app/commonocean-sim/src
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py
python3 /app/usv-navigation/commonocean_integration/scripts/commonocean_scenario.py <path.xml>
```

The first planning problem is controlled by the COLAV automaton; remaining vessels and
dynamic obstacles are handled by commonocean-sim defaults.

> **Note:** Traffic trajectories are automatically interpolated from the scenario's 10s
> timestep to the simulation's 1s timestep so both vessels use the same physical time rate.

## Batch evaluation

Evaluate the automaton across a large set of CommonOcean XML scenarios:

```bash
cd /app/commonocean-sim/src

# Run all scenarios
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py

# Quick test with a small subset
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py --limit 10

# Resume a previous run (skips already-completed scenarios)
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py --resume

# Custom scenarios directory and output
python3 /app/usv-navigation/commonocean_integration/scripts/batch_evaluate.py \
    --scenarios-dir /app/scenarios \
    --output-dir /app/usv-navigation/output/batch_eval \
    --limit 50 --start 0
```

There are three batch runners, one per dataset (they differ in how traffic trajectories
are sourced): `batch_evaluate.py` (generic), `batch_evaluate_handcrafted.py`
(pre-computed trajectories from the XML), and `batch_evaluate_marine_cadastre.py`
(straight-line traffic synthesized from AIS-derived planning problems). All support
`--limit`, `--start`, `--resume`, and `--max-runtime`; the handcrafted and
MarineCadastre runners additionally support `--scenario-ids`.

Results are saved incrementally to `output/batch_eval/results.csv` with per-scenario
metrics including CPA distance, goal reached, collision detected, encounter type, and
automaton state time distribution. Summary plots are generated on completion.

**Evaluation configuration:** `Cs=300.0`, `tp=3.0`, `a=1.67`, `eta=3.5` (real-world
scale, `dt=1 s`). Headline results are in the
[root README](../README.md#evaluation-results).

## Visualization

- `animate_scenario.py` — render a scenario run as a GIF, with unsafe-set overlay
- `replay_scenario.py` — replay a single scenario by ID with visual display
- `plot_trajectories.py` — trajectory plots for selected scenarios

Plots and animations show the vessel trajectory with state-based colouring (blue S1,
red S2, orange S3), obstacle positions with safety-radius circles, unsafe-set convex
hulls, the virtual waypoint V1 while avoiding, heading arrows, and a live state
indicator with time readout.

## Adapter notes

- The adapter drives the deterministic core via `SyncColavRuntime.step_external`: the
  simulator owns vessel integration, and the adapter feeds pose and obstacle states in
  and applies the returned control each tick — no background thread, no wall-clock
  dependence.
- **Post-avoidance waypoint recovery** lives here, not in the core: in multi-waypoint
  scenarios the adapter skips route waypoints that are behind the vessel after an
  S2/S3 → S1 transition, preventing backtracking after an avoidance manoeuvre
  (see `adapters/controller.py`; the core automaton itself tracks a single goal plus V1).
