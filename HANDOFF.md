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
   has **never** passed reliably under any configuration. Needs a T-1022-style investigation
   (`animate_scenario.py T-838`, transition log). Suspects: residual chattering (2–3
   activations despite hysteresis) and/or V1 side selection in the give-way crossing.
   Artifacts: `output/t838_check/run{1..5}`.
2. **Run-to-run non-determinism** (controller's wall-clock asyncio thread): T-838 outcomes
   above vary across identical runs; concurrent container load also shifts timing-sensitive
   outcomes. The roadmap's phase 4 (tick-synchronous runtime) is the systemic fix — T-838 may
   simply be the scenario that sits closest to a guard boundary.
3. **`C-USA_UWC-1_2019012409` collides** (MarineCadastre, CPA ≈ 62 m) — pre-existing across all
   configurations; avoidance activates but fails. Likely needs G22/threshold tuning, not V1 work.
4. `C-USA_UWC-1_2019011202` has CPA ≈ 104 km and can never reach goal — likely a degenerate
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

**Phase 2 — live AIS replay.** Real ship traffic (aisstream.io websocket, or recorded AIS from
a busy strait) into the controller via a thin adapter beside `commonocean_integration/`.

**Phase 3 — interactive web demo.** Browser sandbox (drag obstacles, watch RI/states/V1 live);
FastAPI+websocket, or Pyodide client-side for GitHub Pages. Shares the AIS-replay frontend.

**Phase 4 — deterministic runtime.** Replace the wall-clock asyncio controller with a
tick-synchronous executive. Small, high leverage: makes every batch number reproducible
(T-1022 proved outcomes currently vary run-to-run) and is the prerequisite for any
verification claim. Natural candidate for targeted C++/Rust if desired.

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

## 5. Operational notes

- Docker stack: `docker/start.sh` (detached; noVNC at http://localhost:6080/vnc.html). The
  `docker-sim` image is already built. Source is volume-mounted — edits apply without rebuild.
- Batch run wall-time: ~10–15 s per Handcrafted scenario, so the full 2000 take ~5–8 h; use
  `--scenario-ids` for spot-checks.
- Baseline CSVs worth keeping: `output/batch_eval_handcrafted{,_old}/results.csv` and
  `output/batch_eval_marine_cadastre{,_old}/results.csv` (the old/new V1 comparison above).
  `output/` is gitignored — do not delete these when cleaning up.
