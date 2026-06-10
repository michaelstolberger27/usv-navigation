# Handoff Notes

Context that isn't obvious from reading the code. For architecture/parameters that *are*
derivable from the source, see the code and `CLAUDE.md`. This file covers the *why* behind
recent in-flight work and what to do next.

Last updated: 2026-06-10. Branch: `Updates` (tracks `origin/Updates`, not `main`).

---

## 1. Current state — clean tree, V1 swept-region change validated and committed

The previously uncommitted work is now committed in slices on `Updates`:

- `f64098c` — **V1 capped swept region** (behavioural, validated — see §2)
- `c7ca57d` — CLAUDE.md / HANDOFF.md tracked; `MS_FYP*.tex` gitignored (deliberate user decision)
- `14b9aac` — `--scenario-ids` filter in both dataset batch scripts; stale docstring path fixed
- `d1b579f` — unsafe-set polygon visualisation (controller tracker + animator) + regenerated GIFs

Nothing is uncommitted. The thesis sources (`MS_FYP.tex`, `MS_FYP_BACKUP.tex`) stay **out of
version control by explicit user choice** (repo may become a public portfolio piece) — they are
in `.gitignore`; do not re-add them.

## 2. V1 swept-region validation (2026-06-10 session)

The HANDOFF written earlier assumed the V1 change was unvalidated. It turned out the full
2000-scenario Handcrafted run in `output/batch_eval_handcrafted/` (2026-04-16) was made *after*
the source edits (2026-04-15, volume-mounted) — i.e. **with the new V1 logic**. A 3-scenario
spot-check (T-46/T-27/T-28) reproduced its numbers within ~1%, confirming the CSV matches the
committed code. Comparison vs `output/batch_eval_handcrafted_old/` (2026-03-31, old V1 logic):

| Metric (2000 Handcrafted scenarios) | old V1 (current-pos only) | new V1 (capped sweep) |
|---|---|---|
| collisions | 16 | **1** |
| goal failures | 21 | **5** |
| CPA < 300 m (of 835 avoidance runs) | 146 | **81** |
| avg CPA when avoiding | 526.8 m | 540.4 m |

MarineCadastre (25 scenarios): same single pre-existing collision in both runs
(`C-USA_UWC-1_2019012409`, CPA ≈ 62 m). The apparent goal-failure increase (3→6) is a
**timeout-budget artifact** — the old run used up to 20000 steps, the new one capped at 5000;
all three new-only failures have `avoidance_activations=0`, so the V1 change cannot be the cause.

## 3. Open issues (ranked)

1. **`ZAM_AAA-1_20240121_T-1022` collides** (Handcrafted, CPA 161.8 m, goal not reached) — the
   single remaining collision. Note: `output/batch_eval_t1022/` actually contains a run of
   **T-102**, not T-1022 (ID substring mix-up), so this scenario has *not* been investigated yet.
   Use `--scenario-ids ZAM_AAA-1_20240121_T-1022` (exact match) and `animate_scenario.py` to see
   what goes wrong.
2. **`C-USA_UWC-1_2019012409` collides** (MarineCadastre, CPA ≈ 62 m) — pre-existing in both V1
   variants; avoidance activates but fails. Likely needs G22/threshold tuning, not V1 work.
3. **4 remaining Handcrafted goal failures** (T-584, T-830, T-838, T-1289) — no collisions, just
   goal not reached; lower priority.
4. `C-USA_UWC-1_2019011202` has CPA ≈ 104 km and can never reach goal — likely a degenerate
   scenario; consider excluding it from MC stats rather than chasing it.

## 4. Next phase: portfolio work (agreed with user, 2026-06-10)

Goal: turn the project into an advanced portfolio piece. Agreed sequence:

1. **Foundation polish** — README with architecture diagram + embedded GIFs, a real test suite
   (none exists), CI, possibly publish `colav-automaton` to PyPI. Note: `examples/output/` is
   gitignored but the six GIFs in it are tracked (added before the ignore rule) — README can
   reference them, but new outputs there won't be tracked.
2. **Live AIS replay** — feed real ship traffic (aisstream.io websocket, or recorded AIS from a
   busy strait) into the controller in place of CommonOcean obstacles. The core's
   simulator-independence makes this a thin new adapter beside `commonocean_integration/`.
3. **Interactive web demo** — browser scenario sandbox sharing the AIS-replay frontend
   (FastAPI+websocket, or Pyodide fully client-side for GitHub Pages hosting).
4. (Optional, if targeting robotics roles) ROS 2 node + VRX simulation.

## 5. Operational notes

- Docker stack: `docker/start.sh` (detached; noVNC at http://localhost:6080/vnc.html). The
  `docker-sim` image is already built. Source is volume-mounted — edits apply without rebuild.
- Batch run wall-time: ~10–15 s per Handcrafted scenario, so the full 2000 take ~5–8 h; use
  `--scenario-ids` for spot-checks.
- Baseline CSVs worth keeping: `output/batch_eval_handcrafted{,_old}/results.csv` and
  `output/batch_eval_marine_cadastre{,_old}/results.csv` (the old/new V1 comparison above).
  `output/` is gitignored — do not delete these when cleaning up.
