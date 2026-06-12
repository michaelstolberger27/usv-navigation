# Handoff Notes

Context that isn't obvious from reading the code. For architecture/parameters that *are*
derivable from the source, see the code and `CLAUDE.md`. This file covers the *why* behind
recent in-flight work and what to do next.

Last updated: 2026-06-11. Branch: `Updates` (tracks `origin/Updates`, not `main`).

---

## 1. Current state — hysteresis validated and committed; roadmap phases 0–1 done

Recent commits on `Updates` (newest first):

- `70192b2` — **K_off risk-index hysteresis** on S3→S1 resume (behavioural, validated — see §2/§3)
- `09ce920` — 74-test pytest suite + GitHub Actions CI + `dev` extra
- `c012cab` — V1 capped swept region (behavioural, validated)
- `10ca631` — CLAUDE.md / HANDOFF.md tracked; `MS_FYP*.tex` gitignored (deliberate user decision)
- `af15ac5` / `de8c793` — `--scenario-ids` filter; unsafe-set visualisation + regenerated GIFs

The thesis sources (`MS_FYP.tex`, `MS_FYP_BACKUP.tex`) stay **out of version control by explicit
user choice** (repo may become a public portfolio piece) — they are in `.gitignore`; do not
re-add them. Local test runs need `PYTHONPATH=src` (see §4 phase 1 note).

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

1. **`ZAM_AAA-1_20240121_T-838` — the remaining Handcrafted failure (flaky, ~60% collision).**
   With hysteresis, 5 reruns: 3 collisions (CPA 103–146 m), 2 passes (CPA ~340–370 m), always
   2–3 avoidance activations. crossing_from_starboard (ego is give-way). History: collided
   under old V1 (CPA 125 m), goal-failure under capped-swept V1 (CPA 446 m), now flaky — it
   has **never** passed reliably under any configuration.
   **Preliminary diagnosis (2026-06-11, from `output/t838_check/*.gif`):** the traffic vessel
   crosses from starboard heading NE; the ego (give-way, should pass *astern*/south) instead
   turns **port and crosses ahead**, skirting the swept region's northern flank — V1 lands on
   the port side because the hull's SW (astern) vertices score poorly on predicted CPA with
   the obstacle's *current* position nearby. Timing then decides collision-or-miss → flaky.
   Candidate fix: COLREGs cross-astern preference in `compute_v1` for crossing_from_starboard
   (the encounter classifier already exists in `conditions.py`). Behavioural — needs a full
   batch validation; consider doing it together with the phase-5 parameter sweeps.
   Artifacts: `output/t838_check/run{1..5}` + GIF.
2. **Run-to-run non-determinism** (controller's wall-clock asyncio thread): T-838 outcomes
   above vary across identical runs; concurrent container load also shifts timing-sensitive
   outcomes. The roadmap's phase 4 (tick-synchronous runtime) is the systemic fix — T-838 may
   simply be the scenario that sits closest to a guard boundary.
3. **Dense-traffic degeneracy (found 2026-06-11 on real Singapore Strait data, 393 vessels).**
   Two independent defects, both rooted in the two-vessel assumptions the system was tuned on:
   a. `compute_unified_unsafe_region` builds ONE convex hull over all obstacles — over ~80
      in-range scattered vessels that hull covers the entire strait (see
      `output/ais_replay/singapore_strait.gif`), so any LOS intersects it: G11 fires
      immediately and ¬G23 can never fire. Needs per-obstacle regions / union-not-hull /
      clustering for guard checks (core geometry change — batch revalidation required).
   b. **Global K_off hysteresis deadlocks S3 in traffic**: resume needs max RI over ALL
      obstacles < K_off, and a busy strait always has someone approaching. The ego crossed
      9 km (~1400 s) frozen in S3, reaching goal only because the held heading pointed there.
      Fix: per-threat hysteresis (resume when the RI of the obstacle(s) that triggered
      avoidance subsides, not the global max).
   Despite both, the transit was safe (min CPA 305 m > Cs) — but state semantics are wrong
   and a goal off the held heading would have failed. Rank this alongside phase 4/5 work.
4. **`C-USA_UWC-1_2019012409` collides** (MarineCadastre, CPA ≈ 62 m) — pre-existing across all
   configurations; avoidance activates but fails. Likely needs G22/threshold tuning, not V1 work.
5. `C-USA_UWC-1_2019011202` has CPA ≈ 104 km and can never reach goal — likely a degenerate
   scenario; consider excluding it from MC stats rather than chasing it.

**RESOLVED — `T-1022` (was issue #1).** Root cause: predictive-trigger (G11∧G22) vs
static-resume (G23) asymmetry caused S2/S3/S1 chattering with V1 recomputed (and side-flipped)
each re-entry; full analysis in `output/t1022_investigation/`. Fixed by the K_off risk-index
hysteresis on resume (commit with this HANDOFF update): T-1022 now passes 5/5 with CPA ≈ 575 m
and near-identical trajectories across runs. Full-batch effect: goal failures 5 → 1,
collisions 1 → 1 (the remaining one being T-838 above, which was already failing).

## 4. Roadmap (agreed with user; last revised 2026-06-10)

Goal: an advanced portfolio piece. User decisions on scope: **no PyPI packaging**, **no
wholesale C/C++ port** (targeted C++ only where it earns its place). User is explicitly
interested in **physical AI and GPU/accelerated programming** — phases 5–7 exist to serve that.

**Phase 0 — DONE (2026-06-11): hysteresis validated.** Full Handcrafted batch
(`output/batch_eval_handcrafted_hysteresis/`): 2000/2000 run, **1 collision, 1999 goals
reached** (pre-hysteresis: 1 collision + 5 goal failures; old V1: 16 + 21). The remaining
failure is T-838 — see §3.1, it predates the hysteresis and has never passed reliably.

**Phase 1 — foundation polish (committed with this update).** 74-test pytest suite (`tests/`),
GitHub Actions CI (`.github/workflows/ci.yml`, installs hybrid-automaton from GitHub since it
is not on PyPI), README refresh (correct guards, embedded GIFs, results, badge). Note:
`examples/output/` is gitignored but the six GIFs are tracked (added before the ignore rule) —
README references them; new outputs there won't be tracked. Local test runs need
`PYTHONPATH=src` (the host's pip-installed `colav_automaton` points at the *sibling*
`Desktop/colav-automaton` repo, not this one).

**Phase 2 — AIS replay (core DONE 2026-06-11; live mode unexercised).** `ais_replay/` package:
lat/lon local frame, per-vessel dead-reckoning tracker (sensor-noise point 1, first layer),
recorded-JSONL replay source + runner + GIF rendering, aisstream.io live client + recorder
(written to the v0 API format but **needs an API key to exercise** — `pip install -e .[ais]`).
Validated end-to-end on the bundled synthetic sample (`ais_replay/sample_data/`, exact
aisstream.io message format): goal reached, single clean starboard avoidance, CPA 1165 m.
Two non-obvious runner facts: (1) the ego model **must clamp yaw rate** (`max_yaw_rate=0.15`)
— control updates arrive every ~2.5 sim s from the 20 Hz wall-clock automaton thread, and
unclamped integration of the prescribed-time u diverges; (2) the `pace` parameter (default
0.02 s/tick) holds the wall:sim ratio in the regime the 2000-scenario batch validated —
running flat-out starves the automaton brain and is only good for smoke tests (phase 4 again).
**Real-data validation (2026-06-11, live aisstream.io key):** 30-min Singapore Strait
recording (`output/ais_replay/singapore_strait.jsonl`, gitignored — keep it; 2432 reports,
393 vessels). Ego crossed the strait N→S: goal reached, min CPA 305 m, but exposed the
dense-traffic degeneracies now ranked as open issue §3.3. Range filtering added to the
tracker (`obstacles_at(near=, within=)`, runner default 8 km) — a strait bbox holds 100+
simultaneous vessels.
**Noise numbers (the EKF decision data):** report intervals median 120 s / p90 360 s
(note: > tracker max_age=300 s, so some tracks flicker — consider raising max_age or making
it speed-dependent). Dead-reckoning prediction error at next report: median 7.6 m,
p90 75 m, p99 573 m, max 2.2 km. The median says CV dead-reckoning is fine; the tail is
*maneuvering during long gaps*, which a CV-EKF would NOT fix (needs IMM/coordinated-turn,
or — cheaper and more robust — uncertainty-inflated margins, noise-plan point 2). Anchored
vessels confirmed broadcasting garbage COG (p90 jump 108°) — harmless, v≈0 nulls it.
**Verdict: plain EKF deprioritised; per-threat hysteresis + non-convex unsafe regions
(issue §3.3) are the real dense-traffic work, and margins-from-uncertainty beats filtering
for the maneuver tail.**

**Phase 3 — interactive web demo.** Browser sandbox (drag obstacles, watch RI/states/V1 live);
FastAPI+websocket, or Pyodide client-side for GitHub Pages. Shares the AIS-replay frontend.

**Phase 4 — deterministic runtime (core DONE 2026-06-11; CommonOcean adapter migration
remains).** `SyncColavRuntime` (`src/colav_automaton/sync_runtime.py`): tick-synchronous
executive sharing the exact guards/resets/flows with the async path (plain implementations
split from the framework decorators — see CLAUDE.md, the split is load-bearing). Prescribed-
time clock runs on sim time since transition. Verified: bit-identical reruns (unit tests +
the real-strait replay), avoid/hold/resume cycle on a scripted head-on, async test suite
untouched. AIS replay runner migrated — the yaw-clamp and pace workarounds are gone (control
recomputes every tick by construction; `pace` is now display-only).
**Equivalence picture:** sparse traffic matches the async results (bundled sample: goal Y,
CPA 1178 m vs async 1162–1165 m). Dense traffic now FAILS deterministically instead of
passing by luck: on the strait recording the ego enters S2, actually steers toward the
degenerate-hull V1 (async never did — its control was stale), then deadlocks in S3 off-goal
(goal N, CPA 68 m, identical every rerun). This is open issue §3.3 reproduced exactly — the
strait JSONL is now the regression fixture for those fixes.
**Adapter migration (DONE 2026-06-12).** `step_external()` added (host owns integration;
returned u is the post-reset buffer value, matching async semantics);
`HybridAutomatonController` now steps the sync runtime once per sim tick — no background
thread. Determinism shown in-sim: T-838 three runs identical to the last decimal; an
8-scenario batch re-invocation reproduced every metric column exactly.
**KEY FINDING — tp_control:** the prescribed-time law's tp=3 s horizon was only ever stable
because the async runtime measured it in *wall* time (≈200 sim steps at batch speed);
evaluated faithfully at dt=1 the singular gain spans 3 samples and destabilised the YP plant
(collisions on T-46/T-584/T-1289). `tp_control` now decouples the control-convergence
horizon from tp (which still sets dsafe/delta). Sweep {30,60,120,200} sim-s on the 8 key
scenarios: **60 wins** — 8/8 goals, 0 collisions, CPAs within metres of the async baseline
(T-27 719/718, T-28 461/461, T-584 707/701, T-830 690/682, T-1022 572/575), and T-838
passes at 314 m (it never passed reliably before). 30 → 3 goal failures; 120 → T-838
collides; 200 → all CPAs degrade. Adapter default: `max(60*dt, tp)`; `--tp-control` sweep
flag on the handcrafted batch script.
**Remaining for phase 4:** full 2000-scenario A/B (sync vs async baseline 1 collision/1999
goals) — RUNNING in the container (`output/batch_eval_handcrafted_sync/`, started
2026-06-12); then re-run MarineCadastre, retire the async path from the adapter docs, and
make sync the documented default everywhere.

**Phase 5 — GPU-vectorized Monte Carlo evaluator.** JAX or NVIDIA Warp re-implementation of
the lightweight parts (kinematics, guards, risk index, V1 geometry) to run thousands of
encounters in parallel. Removes the real bottleneck: full batches take ~8–24 h sequentially,
which is why K/K_off/max_horizon have never been properly swept. Also becomes the RL training
environment for phase 6. Do NOT GPU-accelerate the controller itself — it is microseconds of
work; the win is parallel evaluation, not per-tick speed.

**Phase 6 — physical AI.** In order: (a) learned vessel-trajectory prediction trained on the
MarineCadastre AIS data, replacing constant-velocity prediction in the swept region; (b) safe
RL — policy trained in the phase-5 GPU sim with the automaton's guards as a formal safety
shield (neural policy + formal filter architecture).

**Sensor-noise plan (threads through phases 2/5/6, not its own phase):**
(1) *State estimation* — per-vessel tracking layer (dead-reckoning first, EKF when needed)
lands in **phase 2**: live AIS is sparse/async, so the adapter needs it regardless of noise.
Lives in the adapter layer; the core keeps consuming clean (x, y, v, psi).
(2) *Uncertainty-aware margins* (Cs/dsafe inflated by filter covariance, conservative
DCPA/TCPA bounds) — **phase 5/6 boundary**: needs covariance from (1) and the GPU evaluator
to tune k·σ against Monte Carlo noise sweeps instead of guessing.
(3) *Guard robustness* — the K_off hysteresis (committed with the T-1022 fix) is the core of
it; add dwell times only if the phase-5 noise-injection sweeps show residual guard chattering.
Don't start any of this before phase 2 — CommonOcean inputs are noise-free, so there is
nothing to test against yet.

**Phase 7 — ROS 2 + VRX.** Wrap the core in a ROS 2 node and run in the VRX Gazebo simulator;
write the node in **C++ (rclcpp)** — this is the agreed targeted-C++ entry point, not a port
of the core.

**Phase 8 — verification + publication (can run alongside 5–7).** Runtime safety monitor
(invariant + COLREGs conformance logging over the batch corpus), optionally reachability
analysis (CORA/Flow*/SpaceEx) for representative encounter classes; depends on phase 4.
Convert `MS_FYP.tex` into an arXiv preprint / OCEANS / IFAC CAMS submission — the hysteresis
ablation (16→1→pending-0 collisions) is the publishable core.

**Phase 9 — embedded/hardware capstone (water optional).** Target compute: Jetson Orin Nano
(~$250 dev kit; NOT the EOL 2019 Nano). Credibility ladder, each rung valuable on its own:
(a) HIL — full stack on the Jetson, VRX/desktop sim as the world over real ROS 2 topics;
publish loop-latency/jitter/utilization numbers on target silicon. (b) Real sensors without
a vessel — RTL-SDR AIS receiver (~$30) for live RF ship traffic if near a waterway, GPS+IMU
"walking ego" rig, NMEA serial replay. (c) Embedded inference benchmarks — phase-6 models via
TensorRT on the Jetson (latency/watts). (d) Ground surrogate — RC rover(s) with ArduRover +
Jetson companion, architecturally identical to a USV. (e) On-water demo only if access
materializes — the stack transfers unchanged via the phase-7 ROS 2 node.

## 4b. Repo-hygiene backlog (assessed 2026-06-11, deferred by user choice)

Architecture itself is sound (core/adapter split proven twice); these are hygiene items,
ranked, for a future cleanup session:

1. **Git history bloat** — ~369 MB pack for 53 commits (GIF regeneration churn +
   `refs/original/` filter-branch backup refs). Fix: drop backup refs + `git gc`; optionally
   one history rewrite (force push) to keep only current GIF blobs — do this BEFORE the repo
   gets watchers/forks, after that it's expensive.
2. **Media handling** — `examples/output/` is gitignored yet 7 GIFs are force-tracked inside
   it. Move curated demo media to a tracked `assets/` dir; keep all `output/` ignored. Render
   README GIFs smaller (~7 MB of images currently).
3. **Personal documents in `docs/`** (gitignored): FYP handbook/report/supervisor PDFs live
   inside the repo folder, one `git add -f` from publication. Move outside the repo or into
   an ignored `thesis/` with the `.tex` files.
4. **Batch-script triplication** — `batch_evaluate*.py` ≈ 1340 lines, ~70% shared; extract
   `evaluation/batch_runner.py` + thin entry points. Touches validation tooling — own session.
5. **Script audit** — `replay_scenario.py` vs `animate_scenario.py` overlap;
   `plot_trajectories.py`, `commonocean_collision_test.py` possibly superseded. Keep+document
   or delete.
6. **No LICENSE file** — public portfolio repo currently "all rights reserved"; add MIT +
   `license`/`authors` in pyproject.

## 5. Operational notes

- Docker stack: `docker/start.sh` (detached; noVNC at http://localhost:6080/vnc.html). The
  `docker-sim` image is already built. Source is volume-mounted — edits apply without rebuild.
- Batch run wall-time: ~10–15 s per Handcrafted scenario, so the full 2000 take ~5–8 h; use
  `--scenario-ids` for spot-checks.
- Baseline CSVs worth keeping: `output/batch_eval_handcrafted{,_old}/results.csv` and
  `output/batch_eval_marine_cadastre{,_old}/results.csv` (the old/new V1 comparison above).
  `output/` is gitignored — do not delete these when cleaning up.
