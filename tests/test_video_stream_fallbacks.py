"""video_to_event_stream paths that real video files can't reach.

A fake cv2.VideoCapture drives the branches that depend on capture quirks:
missing FPS metadata, missing POS_MSEC (webcams), capture_settings
forwarding, a source whose very first read fails, and early consumer exit.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import eventify.dvs as dvs_mod
from eventify import video_to_event_stream


def _brightening_frames(n=5, size=16):
    return [
        np.full((size, size, 3), 40 + i * 40, dtype=np.uint8) for i in range(n)
    ]


class FakeCapture:
    """Mimics cv2.VideoCapture for a webcam-like source: no FPS, no POS_MSEC."""

    def __init__(self, source):
        self.source = source
        self.frames = list(_brightening_frames())
        self.set_calls = []
        self.released = False
        self._opened = True

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return 0.0  # no FPS, no POS_MSEC metadata

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    created = []

    def factory(source):
        cap = FakeCapture(source)
        created.append(cap)
        return cap

    monkeypatch.setattr(dvs_mod.cv2, "VideoCapture", factory)
    return created


def test_missing_fps_falls_back_to_30fps_period(fake_capture):
    # File source with no FPS/POS_MSEC metadata: fixed 33.3 ms spacing.
    chunks = list(video_to_event_stream("fake.avi"))

    assert len(chunks) == 4  # N-1 chunks for N=5 frames
    for i, chunk in enumerate(chunks):
        assert len(chunk) > 0
        lo, hi = i * 33_333, (i + 1) * 33_333
        assert np.all(chunk["t"] >= lo)
        assert np.all(chunk["t"] < hi)


def test_live_source_uses_wall_clock(fake_capture, monkeypatch):
    # A fake clock that advances 100 ms per call: t0 lands at 0.1, the four
    # frame pairs at 0.2..0.5, so chunks span 100 ms each, not 33.3 ms.
    ticks = iter(x * 0.1 for x in range(1, 100))
    monkeypatch.setattr(dvs_mod.time, "monotonic", lambda: next(ticks))

    chunks = list(video_to_event_stream(0))

    assert len(chunks) == 4
    assert chunks[0]["t"].max() < 100_000
    assert np.all(chunks[1]["t"] >= 100_000)
    assert chunks[-1]["t"].max() < 400_000


def test_warmup_frames_are_discarded(fake_capture):
    # 5 frames, 2 burned on warmup: one reference frame + 2 pairs remain.
    chunks = list(video_to_event_stream("fake.avi", warmup_frames=2))
    assert len(chunks) == 2


def test_duration_limits_a_file_stream(fake_capture):
    # Fallback timestamps run at 33.3 ms per pair; a 40 ms budget covers the
    # first pair fully and stops after the second crosses the limit.
    chunks = list(video_to_event_stream("fake.avi", duration_s=0.04))
    assert len(chunks) == 2
    assert fake_capture[0].released


def test_duration_stops_a_live_capture(fake_capture, monkeypatch):
    ticks = iter(x * 0.1 for x in range(1, 100))
    monkeypatch.setattr(dvs_mod.time, "monotonic", lambda: next(ticks))

    # Pairs land at 100/200/300 ms; a 250 ms budget stops after the third.
    chunks = list(video_to_event_stream(0, duration_s=0.25))
    assert len(chunks) == 3
    assert fake_capture[0].released


def test_capture_settings_are_forwarded(fake_capture):
    settings = {cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FPS: 15}
    list(video_to_event_stream(0, capture_settings=settings))

    assert fake_capture[0].set_calls == list(settings.items())


def test_int_source_is_passed_through(fake_capture):
    list(video_to_event_stream(3))
    assert fake_capture[0].source == 3


def test_first_read_failure_yields_nothing_and_releases(monkeypatch):
    created = []

    def no_frames(source):
        cap = FakeCapture(source)
        cap.frames = []
        created.append(cap)
        return cap

    monkeypatch.setattr(dvs_mod.cv2, "VideoCapture", no_frames)

    chunks = list(video_to_event_stream(0))
    assert chunks == []
    assert created[0].released


def test_capture_released_when_consumer_stops_early(fake_capture):
    gen = video_to_event_stream(0)
    next(gen)
    gen.close()  # consumer walks away mid-stream

    assert fake_capture[0].released


def test_unopenable_source_raises_and_releases(monkeypatch):
    class ClosedCapture(FakeCapture):
        def __init__(self, source):
            super().__init__(source)
            self._opened = False

    created = []

    def factory(source):
        cap = ClosedCapture(source)
        created.append(cap)
        return cap

    monkeypatch.setattr(dvs_mod.cv2, "VideoCapture", factory)

    with pytest.raises(IOError):
        list(video_to_event_stream(0))
    assert created[0].released
