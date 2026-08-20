"""frame_stream edge cases: non-square sensors, short videos, still prefixes."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from eventify import frame_stream


def _write_video(path, values, size=(64, 48)):
    """Write one uniform frame per value; size is (W, H) as cv2 expects."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h), isColor=True)
    assert writer.isOpened()
    for v in values:
        writer.write(np.full((h, w, 3), v, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def brightening_video(tmp_path):
    return _write_video(tmp_path / "bright.avi", [30 + i * 30 for i in range(8)])


def test_nonsquare_sensor_size_is_width_height(brightening_video):
    # sensor_size=(W=48, H=32) must yield (n_bins, 2, H=32, W=48) tensors.
    frames = list(
        frame_stream(str(brightening_video), sensor_size=(48, 32), n_bins=2,
                     window_ms=200)
    )
    assert len(frames) > 0
    for f in frames:
        assert f.shape == (2, 2, 32, 48)


def test_video_shorter_than_window_yields_nothing(brightening_video):
    # 8 frames at 10 fps ≈ 0.7 s of events; a 10 s window never fills.
    frames = list(
        frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4,
                     window_ms=10_000)
    )
    assert frames == []


def test_still_prefix_chunks_are_skipped(tmp_path):
    # Three identical frames produce empty chunks before motion starts; the
    # window clock must start at the first event, not at t=0.
    video = _write_video(
        tmp_path / "still_prefix.avi", [40, 40, 40, 80, 120, 160, 200, 240]
    )
    frames = list(
        frame_stream(str(video), sensor_size=(32, 32), n_bins=2, window_ms=200)
    )
    assert len(frames) > 0
    # Every yielded window overlaps actual motion, so none is all-zero.
    for f in frames:
        assert f.sum() > 0
