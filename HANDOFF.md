# Handoff Notes

Context that isn't obvious from reading the code. For architecture/parameters that *are*
derivable from the source, see the code and the project memory index. This file covers the
*why* behind recent in-flight work and what to do next.

Last updated: 2026-06-10. Branch: `Updates` (tracks `origin/Updates`, not `main`).

---

## 1. Current state — what's uncommitted and why it matters

`git status` shows a working tree with **uncommitted changes that have NOT been re-validated by a
batch run**. Do not assume the "60/60 goal, 0 collisions" results in memory still hold — those were
produced with the *previous* V1 logic (see §3). The current diff changes V1 placement materially.

Three logical changes are intermixed in the working tree:

1. **V1 swept-region reversal** (`resets.py`, `unsafe_sets.py`) — behavioural, needs re-evaluation.
2. **Unsafe-set visualisation** (`controller.py`, `animate_scenario.py`) — cosmetic/diagnostic, safe.
3. **`--scenario-ids` filter** (both batch scripts) — additive, safe.

If you need to commit, **split these**. The visualisation and `--scenario-ids` changes are
low-risk and can go in immediately. The V1 change should be committed only *after* a batch run
confirms it doesn't regress the collision numbers.

---

## 2. The three batch scripts (memory only mentions one)

Memory references a single `batch_evaluate.py`. There are now **three**, one per CommonOcean
dataset, differing in how dynamic-obstacle trajectories are obtained:

| Script | Dataset | Obstacle trajectories |
|---|---|---|
| `batch_evaluate.py` | generic / original | as-provided |
| `batch_evaluate_handcrafted.py` | HandcraftedTwoVesselEncounters | **pre-computed** in the XML (1 ego + 1 traffic) |
| `batch_evaluate_marine_cadastre.py` | MarineCadastre | **synthesized** straight-line constant-velocity from planning problems (multi-vessel only) |

This distinction is the reason the V1 / swept-region logic matters differently per dataset:
Handcrafted obstacles can turn (real trajectories), MarineCadastre obstacles are straight lines.
A `max_horizon` that works for one may misbehave on the other — validate on both.

`batch_evaluate.py`'s docstring still has the **stale `commonocean/scripts/` path** (old dir name).
The real path is `commonocean_integration/scripts/`. Harmless but worth fixing when you touch it.

---

## 3. KEY DECISION REVERSED: V1 now uses the swept region again (capped)

This is the most important thing to understand before continuing.

**History:** An earlier fix (memory item "V1 placement", and the now-deleted comment in `resets.py`)
deliberately set `use_swept_region=False` for V1 computation, with the reasoning: *"the swept region
would place V1 far along the obstacle's trajectory, making it unreachable."* The ship steered around
where the obstacle *is*, not where it *will be*.

**Current uncommitted change reverses this** back to `use_swept_region=True`, but adds a
`max_horizon` cap to fix the original "unreachable V1" problem instead of disabling the sweep:

```python
max_horizon = max(60.0, 3.0 * dsafe / v)   # resets.py, reset_enter_avoidance
```

Rationale: full TCPA can be 500+ s for slow obstacles, which is what made V1 unreachable before.
Capping the sweep at ~3× braking distance keeps V1 reachable while still clearing the obstacle's
*near-future* path rather than only its current position. The cap threads through
`get_unsafe_set_vertices(..., max_horizon=...)` → `_compute_swept_obstacles(..., max_horizon=...)`
where it becomes `sweep_horizon = min(tcpa, max_horizon)`.

**Why this needs attention:** this directly trades off the two failure modes that have flip-flopped
in this project — V1 unreachable (ship never completes avoidance) vs. V1 too close (collision because
it steers around the obstacle's *current* spot while the obstacle moves into the gap). The `max(60.0, ...)`
floor and the `3.0` multiplier are **unvalidated tuning knobs**. The first thing the next session
should do is sweep these on the Handcrafted set (which has real curving trajectories that expose the
difference) and confirm collisions stay at 0.

**Do not delete the `max_horizon` plumbing** if you revert the behaviour — set `use_swept_region=False`
at the call site instead. The plumbing is correct and reusable.

---

## 4. Unsafe-set visualisation (controller + animator)

The controller now records `unsafe_set_tracker` — one entry per step — by calling
`compute_unified_unsafe_region(...)` and storing the polygon exterior coords (or `None`).

Two non-obvious conventions here:

- **It is best-effort.** The whole block is wrapped in `try/except` and appends `None` on any failure,
  with no logging. This is intentional: visualisation must never break or slow a simulation run. If the
  unsafe-set polygon stops appearing in GIFs, the *simulation* is fine — look for an exception being
  swallowed in `controller.py`, not a logic bug.
- **`compute_unified_unsafe_region` vs `get_unsafe_set_vertices`** are two different entry points to the
  same `colav_unsafe_set` API and must not be conflated:
  - `get_unsafe_set_vertices` → returns raw hull *vertices*, used for G11 guards and V1 computation.
    Has the `use_swept_region` / `max_horizon` knobs.
  - `compute_unified_unsafe_region` → returns a shapely `Polygon`, used for **guard checks and
    visualisation only**. Has a `static_only` flag (zero-velocity obstacles) used by G23 resume checks
    because full TCPA prediction is too conservative there and would stop the ship from ever resuming.

`animate_scenario.py` replaced the old CPA line/text overlay with the purple unsafe-set polygon (removed
all the `min_cpa` tracking). It also gained `--xlim/--ylim` for **fixed plot bounds** — needed when
producing comparable side-by-side GIFs across scenarios (auto-bounds make each GIF a different scale).
The committed `examples/output/*.gif` were regenerated with this change (smaller file sizes in the diff).

---

## 5. The `.tex` thesis files (untracked, in repo root)

`MS_FYP.tex` (1672 lines) and `MS_FYP_BACKUP.tex` are the **FYP / thesis writeup** — the academic
deliverable. They are untracked (not gitignored). Decide deliberately whether these belong in version
control; if they should be tracked, do it explicitly, and consider a `figures/`-style structure rather
than a manually-maintained `_BACKUP` copy. **Do not `git add -A` blindly** — you'd commit both the
thesis source and the backup. If they should stay out, add them to `.gitignore`.

The thesis is the source of truth for the paper equation references sprinkled through the code
("paper eq 14", "eq 19-21", etc.). When code and "paper eq N" disagree, the thesis wins — check it.

---

## 6. Conventions established in this codebase

- **`commonocean_integration/`** (not `commonocean/`) and **`adapters/controller.py`** (not
  `commonocean.py`) — both names deliberately avoid colliding with the pip-installed `commonocean`
  package. Never rename toward the shorter forms.
- **Paper-equation comments are load-bearing.** Comments like `# paper eq 27` / `# paper eq 14` are how
  the implementation stays traceable to the thesis. Keep them when editing; update them if the math changes.
- **API-failure fallbacks are deliberate, not defensive boilerplate.** `get_unsafe_set_vertices` and
  `compute_unified_unsafe_region` both return `None` and (sometimes) log a warning on
  `create_unsafe_set` failure; callers have explicit fallback paths (e.g. `default_vertex_provider`).
  The underlying `colav_unsafe_set` API throws on degenerate geometry — these aren't paranoia.
- **Trackers are parallel lists indexed by step**: `position_tracker`, `state_tracker`, `v1_tracker`,
  `unsafe_set_tracker` all advance one entry per controller tick and must stay length-aligned. If you
  add an early-return path in the tick, append to *all* trackers or the animation desyncs.
- **`max(floor, formula)` tuning idiom**: parameters like `max_horizon`, `v1_dist`, TCPA thresholds all
  use `max(<floor>, <v-scaled formula>)`. The floor prevents degenerate behaviour at low speed. When
  tuning, change the multiplier first, the floor second.

---

## 7. Specific next steps

1. **Re-validate the V1 swept-region change (§3).** Run `batch_evaluate_handcrafted.py` (real curving
   trajectories expose the regression) and confirm 0 collisions. Then MarineCadastre. Only commit the
   `resets.py`/`unsafe_sets.py` changes after this passes. Use the new `--scenario-ids` to spot-check
   the three historically-tight scenarios first (from memory: T-46, T-27, T-28).
2. **Sweep `max_horizon` knobs** if step 1 regresses: the `3.0 * dsafe / v` multiplier and `60.0` floor.
3. **Commit in three slices** (§1) so the behavioural change is bisectable separately from cosmetics.
4. **Decide the `.tex` files' fate** (§5) before any `git add`.
5. **Fix the stale docstring path** in `batch_evaluate.py` (§2) when convenient.
6. **Update project memory** after re-validation — the "60/60, 0 collisions" numbers and the "V1 uses
   current position only" note are both now stale.

---

## 8. Files needing attention

| File | Why |
|---|---|
| `src/colav_automaton/resets/resets.py` | V1 swept-region reversal — unvalidated, highest risk (§3) |
| `src/colav_automaton/controllers/unsafe_sets.py` | new `max_horizon` plumbing; two similar entry points easy to confuse (§4) |
| `commonocean_integration/scripts/batch_evaluate.py` | stale docstring path; oldest of the three scripts (§2) |
| `MS_FYP.tex`, `MS_FYP_BACKUP.tex` | untracked thesis + backup; VC decision needed (§5) |
| `examples/output/*.gif` | regenerated; verify they render the unsafe-set polygon before committing |
