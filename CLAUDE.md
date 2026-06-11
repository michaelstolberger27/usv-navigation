# CLAUDE.md

Hybrid-automaton-based, COLREGs-compliant collision avoidance for USVs. This file holds
**lasting conventions**. For in-flight work and unvalidated changes, see `HANDOFF.md`.

## Architecture

- **`src/colav_automaton/`** — the core automaton. **No simulator dependencies.** Depends only on
  `numpy`, `shapely`, `hybrid-automaton`, `colav-unsafe-set`. Keep it that way: nothing CommonOcean
  or pyglet may be imported here.
- **`commonocean_integration/`** — all simulator-specific code (adapters, evaluation, scripts).
  Anything that imports `commonocean`/`commonroad`/the sim lives here, never in `src/`.
- **3-state hybrid automaton** (`automaton.py`, `guards/`, `resets/`, `invariants/`): S1 waypoint-reaching,
  S2 collision-avoidance (steers to virtual waypoint V1), S3 constant-control (hold heading until LOS clears).
  Transitions: S1→S2 `G11 ∧ G22`, S2→S3 `¬L1 ∨ ¬L2`, S3→S1 `¬G23 ∧ RI<K_off`.
- **Two runtimes, one automaton.** `ColavAutomaton` (async, wall-clock; CommonOcean batches were
  validated on it) and `SyncColavRuntime` (`sync_runtime.py`, deterministic tick-synchronous —
  bit-identical reruns; prefer it for new integrations). To make both possible, `dynamics/` and
  `resets/` define **plain functions** (`*_flow`, `apply_*`) with the framework decorators applied
  explicitly at the module bottom — do not merge them back into `@decorator` form (the decorators
  type-check the async framework's Context and would lock out the sync runtime). Guards are
  called via `.func` by the sync executive.
- **`colav_unsafe_set` API has two entry points** in `controllers/unsafe_sets.py` — do not conflate:
  - `get_unsafe_set_vertices` → hull *vertices*; for G11 guards and V1. Has `use_swept_region`/`max_horizon`.
  - `compute_unified_unsafe_region` → shapely `Polygon`; for guard checks and visualisation. Has
    `static_only` (used by G23 resume checks — full TCPA prediction is too conservative there).
- **Step-indexed parallel trackers** in `adapters/controller.py` (`position_tracker`, `state_tracker`,
  `v1_tracker`, `unsafe_set_tracker`) advance one entry per tick and **must stay length-aligned**.
  Any early-return in the tick must append to *all* of them or the animation desyncs.

## Naming (deliberate — do not "simplify")

- `commonocean_integration/` (not `commonocean/`) and `adapters/controller.py` (not `commonocean.py`):
  both avoid colliding with the pip-installed `commonocean` package.
- Package is `colav_automaton` under a **src-layout** (`pyproject.toml` → `where = ["src"]`).

## Build / run / test

- **Install (core, no Docker):** `pip install -e .[viz]` — `[viz]` adds matplotlib for the standalone examples.
- **Standalone examples (no Docker):** `python examples/realtime_simulation.py [--scenario N | --all | --no-unsafe]`.
- **Full sim stack (CommonOcean + Gurobi + VNC) needs Docker:**
  - `docker/start.sh -it` (interactive) or `docker/start.sh` (detached; noVNC at http://localhost:6080/vnc.html).
  - **Sibling-repo layout is required:** `../hybrid-automaton`, `../unsafe-set`, `../commonocean-sim`,
    `../commonocean-scenarios` must sit beside this repo (the compose `additional_contexts` and scenario
    volume mount depend on it).
  - Inside the container, scripts run from `/app/commonocean-sim/src` with
    `PYTHONPATH=/app/commonocean-sim/src:/app/usv-navigation`. Source is volume-mounted (live reload).
  - **Three batch scripts, one per dataset** (differ in how obstacle trajectories are sourced):
    `batch_evaluate.py` (generic), `batch_evaluate_handcrafted.py` (pre-computed trajectories),
    `batch_evaluate_marine_cadastre.py` (synthesized straight-line). Common flags: `--limit`, `--start`,
    `--scenario-ids`, `--resume`, `--max-runtime`.
- **No automated test suite / no linter configured.** `commonocean_collision_test.py` is a manual
  integration check, not pytest. Validate behavioural changes with a batch run, not unit tests.

## Rules for future sessions

- **Never add simulator imports to `src/colav_automaton/`** — the core/integration split is load-bearing.
- **Paper-equation comments are load-bearing.** `# paper eq N` ties code to the thesis (`MS_FYP.tex`).
  Keep them on edits; update them when the math changes. When code and "paper eq N" disagree, the thesis wins.
- **`colav_unsafe_set` throws on degenerate geometry.** The `None`-returning fallbacks in `unsafe_sets.py`
  and their caller-side fallback paths are deliberate, not removable boilerplate.
- **Tuning idiom is `max(<floor>, <v-scaled formula>)`** (V1 distance, `max_horizon`, TCPA thresholds).
  The floor guards low-speed degeneracy. When tuning, change the multiplier first, the floor second.
- **Validate behavioural changes with a batch run before committing** — prefer the Handcrafted set
  (real curving trajectories expose V1 regressions). Commit cosmetic and behavioural changes separately.
- **The work branch is `Updates`, not `main`.** `MS_FYP.tex`/`MS_FYP_BACKUP.tex` are untracked thesis
  files — never `git add -A` blindly.
- Response style: concise, no emojis, markdown links for file references.
