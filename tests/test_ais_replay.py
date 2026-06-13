"""
Tests for the AIS replay adapter: projection, unit conversions, the
dead-reckoning tracker, and recorded-source parsing.

Network-dependent parts (AISStreamSource) are exercised manually, not
here.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ais_replay.geo import LocalFrame, M_PER_DEG_LAT, cog_to_psi, knots_to_ms
from ais_replay.runner import ReplayRunner
from ais_replay.sources import RecordedAISSource, parse_aisstream_message
from ais_replay.tracker import AISReport, TrafficTracker

SAMPLE = Path(__file__).parent.parent / "ais_replay" / "sample_data" / "sample_strait.jsonl"


class TestLocalFrame:
    def test_round_trip(self):
        frame = LocalFrame(1.2, 103.85)
        lat, lon = frame.to_latlon(*frame.to_xy(1.25, 103.9))
        assert lat == pytest.approx(1.25, abs=1e-9)
        assert lon == pytest.approx(103.9, abs=1e-9)

    def test_does_not_wrap_antimeridian(self):
        # Documents a known limitation: the equirectangular projection is
        # not antimeridian-aware. Two points either side of +/-180 lon are
        # ~0.02 deg apart in reality but project ~40000 km apart here.
        # The adapter targets bounded harbour/strait bboxes, never the
        # dateline; revisit if an operating area straddles +/-180.
        frame = LocalFrame(0.0, 179.99)
        x_east, _ = frame.to_xy(0.0, -179.99)
        assert abs(x_east) > 3.9e7  # no wrap: ~40000 km, not ~2 km

    def test_one_degree_latitude(self):
        frame = LocalFrame(1.2, 103.85)
        x, y = frame.to_xy(2.2, 103.85)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(M_PER_DEG_LAT)

    def test_longitude_shrinks_with_latitude(self):
        x_equator, _ = LocalFrame(0.0, 0.0).to_xy(0.0, 1.0)
        x_60n, _ = LocalFrame(60.0, 0.0).to_xy(60.0, 1.0)
        assert x_60n == pytest.approx(x_equator * 0.5, rel=1e-3)


class TestConversions:
    def test_cog_north_is_plus_y(self):
        psi = cog_to_psi(0.0)
        assert np.cos(psi) == pytest.approx(0.0, abs=1e-12)
        assert np.sin(psi) == pytest.approx(1.0)

    def test_cog_east_is_plus_x(self):
        psi = cog_to_psi(90.0)
        assert np.cos(psi) == pytest.approx(1.0)
        assert np.sin(psi) == pytest.approx(0.0, abs=1e-12)

    def test_knots(self):
        assert knots_to_ms(10.0) == pytest.approx(5.14444)


def _report(mmsi=1, t=0.0, lat=1.2, lon=103.85, sog=10.0, cog=90.0):
    return AISReport(mmsi=mmsi, t=t, lat=lat, lon=lon,
                     sog_knots=sog, cog_deg=cog)


class TestTrafficTracker:
    def setup_method(self):
        self.frame = LocalFrame(1.2, 103.85)
        self.tracker = TrafficTracker(self.frame, max_age=300.0)

    def test_dead_reckons_constant_velocity(self):
        # 10 kn eastbound: 60 s later the track is ~308.7 m east.
        self.tracker.ingest(_report(t=0.0))
        (x, y, v, psi), = self.tracker.obstacles_at(60.0)
        assert x == pytest.approx(knots_to_ms(10.0) * 60.0, rel=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)
        assert v == pytest.approx(knots_to_ms(10.0))

    def test_new_report_resets_track(self):
        self.tracker.ingest(_report(t=0.0))
        self.tracker.ingest(_report(t=60.0, lat=1.21))
        (x, y, _, _), = self.tracker.obstacles_at(60.0)
        assert y == pytest.approx(self.frame.to_xy(1.21, 103.85)[1])

    def test_out_of_order_report_ignored(self):
        self.tracker.ingest(_report(t=60.0))
        self.tracker.ingest(_report(t=0.0, lat=1.5))
        (x, y, _, _), = self.tracker.obstacles_at(60.0)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_stale_track_expires(self):
        self.tracker.ingest(_report(t=0.0))
        assert len(self.tracker.obstacles_at(299.0)) == 1
        assert len(self.tracker.obstacles_at(301.0)) == 0
        assert len(self.tracker) == 0

    def test_unavailable_sog_treated_as_stationary(self):
        self.tracker.ingest(_report(t=0.0, sog=102.3))
        (x, y, v, _), = self.tracker.obstacles_at(100.0)
        assert v == 0.0
        assert x == pytest.approx(0.0, abs=1e-6)

    def test_multiple_vessels(self):
        self.tracker.ingest(_report(mmsi=1, t=0.0))
        self.tracker.ingest(_report(mmsi=2, t=0.0, lat=1.21, sog=0.0))
        assert len(self.tracker.obstacles_at(10.0)) == 2

    def test_range_filter(self):
        # mmsi=1 at origin, mmsi=2 ~1113 m north: a 500 m filter around
        # the origin keeps only the first; the filtered track survives.
        self.tracker.ingest(_report(mmsi=1, t=0.0, sog=0.0))
        self.tracker.ingest(_report(mmsi=2, t=0.0, lat=1.21, sog=0.0))
        close = self.tracker.obstacles_at(0.0, near=(0.0, 0.0), within=500.0)
        assert len(close) == 1
        assert len(self.tracker.obstacles_at(0.0)) == 2

    def test_range_filter_uses_dead_reckoned_position(self):
        # Eastbound at 10 kn from the origin: by t=250 s the
        # dead-reckoned position (~1286 m east) has left a 1 km disc
        # around the origin even though no new report arrived.
        self.tracker.ingest(_report(mmsi=1, t=0.0, sog=10.0, cog=90.0))
        assert len(self.tracker.obstacles_at(0.0, near=(0.0, 0.0),
                                             within=1000.0)) == 1
        assert len(self.tracker.obstacles_at(250.0, near=(0.0, 0.0),
                                             within=1000.0)) == 0


class TestParseAisstreamMessage:
    def test_non_position_report_skipped(self):
        assert parse_aisstream_message({"MessageType": "ShipStaticData"}) is None

    def test_parses_sample_line(self):
        with open(SAMPLE) as f:
            msg = json.loads(f.readline())
        report = parse_aisstream_message(msg)
        assert report is not None
        assert report.mmsi == 563012340
        assert report.name == "LADY MARGAUX"
        assert report.sog_knots > 0
        # 2024-01-21 07:00:00 UTC
        assert report.t == pytest.approx(1705820400.0)


class TestRecordedAISSource:
    def test_loads_sample_sorted(self):
        source = RecordedAISSource(str(SAMPLE))
        times = [r.t for r in source.reports]
        assert times == sorted(times)
        assert source.t_end - source.t_start == pytest.approx(1800.0)

    def test_feed_until_advances_cursor(self):
        source = RecordedAISSource(str(SAMPLE))
        tracker = TrafficTracker(LocalFrame(1.2, 103.85))
        fed_first = source.feed_until(tracker, source.t_start + 60.0)
        assert fed_first > 0
        assert len(tracker) == 3  # all three sample vessels seen
        # Feeding the same window again is a no-op
        assert source.feed_until(tracker, source.t_start + 60.0) == 0


class TestReplayRunner:
    """End-to-end smoke test: the automaton driven through recorded AIS."""

    def _run(self):
        frame = LocalFrame(1.2, 103.85)
        source = RecordedAISSource(str(SAMPLE))
        tracker = TrafficTracker(frame)
        start = frame.to_xy(1.2000, 103.8500)
        goal = frame.to_xy(1.2000, 103.8880)
        psi0 = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
        runner = ReplayRunner(
            source, tracker,
            ego_start=(start[0], start[1], psi0), goal=goal,
            v=6.0, Cs=300.0, tp=3.0, pace=0.0, max_duration=1800.0,
        )
        summary = runner.run(verbose=False)
        return runner, summary

    def test_completes_and_trackers_aligned(self):
        runner, summary = self._run()
        n = len(runner.times)
        assert n > 0
        # All step-indexed trackers advance one entry per tick.
        assert len(runner.position_tracker) == n
        assert len(runner.state_tracker) == n
        assert summary["steps"] == n

    def test_deterministic(self):
        # The sync runtime is the whole point: identical recorded inputs
        # must give a bit-identical trajectory (the async runner needed
        # yaw-clamp + pacing hacks to even approximate this).
        r1, s1 = self._run()
        r2, s2 = self._run()
        assert np.array_equal(np.array(r1.position_tracker),
                              np.array(r2.position_tracker))
        assert s1 == s2
