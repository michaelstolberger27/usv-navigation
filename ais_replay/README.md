# AIS replay

Feeds **real AIS vessel traffic** into the COLAV automaton — recorded or live from
[aisstream.io](https://aisstream.io) — with no simulator required. The runner drives
the deterministic `SyncColavRuntime`, so a given recording replays **bit-identically**
every time.

```bash
# From the repo root: replay the bundled sample scenario (Singapore Strait geometry)
PYTHONPATH=src:. python3 ais_replay/scripts/run_replay.py
```

<p align="center">
  <img src="../assets/ais_replay_sample_strait.gif" alt="AIS replay through strait traffic" width="70%"/>
</p>

## Recording and replaying real traffic

Record live traffic with a free [aisstream.io](https://aisstream.io) API key, then
replay it with an ego route of your choosing:

```bash
PYTHONPATH=src:. python3 ais_replay/scripts/record_ais.py \
    --bbox 1.15,103.7,1.35,104.1 --duration 1800 --out strait.jsonl
PYTHONPATH=src:. python3 ais_replay/scripts/run_replay.py \
    --recording strait.jsonl --ego-start 1.20,103.85 --goal 1.20,103.95
```

Recordings are raw aisstream.io JSONL — no preprocessing step. Live mode
(`AISStreamSource`) needs the websocket extra: `pip install -e .[ais]`.

## The tracking layer

AIS reports arrive sparsely and irregularly (2-30+ seconds between position reports
per vessel), while the automaton expects a clean obstacle state for every vessel on
every tick. A per-vessel tracking layer bridges the gap:

- each vessel's latest report (position, speed, course) is **dead-reckoned** forward
  between updates;
- tracks that stop reporting are **expired** so ghost vessels don't trigger avoidance;
- lat/lon is converted to the local metric frame the automaton works in (`geo.py`).

## What replaying real traffic surfaced

Running the automaton against a 30-minute recording of Singapore Strait traffic
(393 vessels) exposed two genuine design limitations — the unified-hull degeneracy in
dense traffic and the global resume-hysteresis freeze. Both are documented in the
root README's [Known limitations](../README.md#known-limitations) and pinned as
`strict` xfail tests in
[`tests/test_behaviour_regression.py`](../tests/test_behaviour_regression.py), so the
suite flags the moment a fix lands.

## Files

| Path | Purpose |
|---|---|
| `geo.py` | lat/lon ↔ local metric frame conversion |
| `tracker.py` | Per-vessel dead-reckoning tracker (sparse AIS → per-tick states) |
| `sources.py` | `RecordedAISSource` (JSONL), `AISStreamSource` (live websocket) |
| `runner.py` | Drives the automaton through AIS traffic via `SyncColavRuntime` |
| `sample_data/` | Bundled synthetic recording (aisstream.io format) |
| `scripts/` | `run_replay.py`, `record_ais.py` |
