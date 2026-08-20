"""CLI internals the roundtrip tests don't reach.

Covers _accum_to_bgr, _ThreadedCapture, the convert command's writer-failure
path, exporting a motionless video, and a fully stubbed run of the live
webcam command (GUI and capture replaced so it works headless, as in CI).
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")

from typer.testing import CliRunner

import eventify.cli as cli_mod
from eventify.cli import _accum_to_bgr, _ThreadedCapture, app

runner = CliRunner()


@pytest.fixture
def synthetic_video(tmp_path):
    """5-frame 32x32 video, each frame uniformly brighter than the last."""
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((32, 32, 3), 40 + i * 40, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def static_video(tmp_path):
    """5 identical frames: a video with no motion at all."""
    path = tmp_path / "static.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((32, 32, 3), 120, dtype=np.uint8))
    writer.release()
    return path


# ---- _accum_to_bgr -----------------------------------------------------------


def test_accum_to_bgr_positive_maps_to_on_color():
    accum = np.full((2, 2), 4.0, dtype=np.float32)
    img = _accum_to_bgr(accum, max_events=8.0)
    # Half intensity of the ON color (BGR 180, 70, 0).
    assert img.dtype == np.uint8
    np.testing.assert_array_equal(img[0, 0], [90, 35, 0])


def test_accum_to_bgr_negative_maps_to_off_color():
    accum = np.full((2, 2), -8.0, dtype=np.float32)
    img = _accum_to_bgr(accum, max_events=8.0)
    # Full intensity of the OFF color (BGR 0, 170, 220).
    np.testing.assert_array_equal(img[0, 0], [0, 170, 220])


def test_accum_to_bgr_zero_is_black_and_overflow_saturates():
    accum = np.array([[0.0, 1000.0]], dtype=np.float32)
    img = _accum_to_bgr(accum, max_events=8.0)
    np.testing.assert_array_equal(img[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(img[0, 1], [180, 70, 0])  # clipped, not wrapped


# ---- _ThreadedCapture ---------------------------------------------------------


def test_threaded_capture_reads_frames_from_file(synthetic_video):
    cap = _ThreadedCapture(str(synthetic_video))
    try:
        frame = cap.read(timeout=2.0)
        assert frame is not None
        assert frame.shape == (32, 32, 3)
        assert cap.actual_size == (32, 32)
    finally:
        cap.release()


def test_threaded_capture_returns_none_when_exhausted(synthetic_video):
    cap = _ThreadedCapture(str(synthetic_video))
    try:
        # Drain: the file has 5 frames; stale ones may be dropped by design,
        # so just read until the stream reports exhaustion.
        for _ in range(20):
            if cap.read(timeout=0.5) is None:
                break
        assert cap.read(timeout=0.2) is None
    finally:
        cap.release()


def test_threaded_capture_unopenable_source_raises(tmp_path):
    with pytest.raises(IOError):
        _ThreadedCapture(str(tmp_path / "missing.avi"))


def test_threaded_capture_release_is_idempotent(synthetic_video):
    cap = _ThreadedCapture(str(synthetic_video))
    cap.release()
    cap.release()  # second call must not raise or hang


# ---- convert: writer failure ---------------------------------------------------


def test_convert_unwritable_output_returns_error(synthetic_video, tmp_path):
    out = tmp_path / "no_such_dir" / "out.mp4"
    result = runner.invoke(app, ["convert", str(synthetic_video), str(out)])
    assert result.exit_code == 1


# ---- export: motionless input ---------------------------------------------------


def test_export_static_video_writes_empty_but_valid_hdf5(static_video, tmp_path):
    out = tmp_path / "events.h5"
    result = runner.invoke(app, ["export", str(static_video), str(out)])
    assert result.exit_code == 0

    with h5py.File(out, "r") as f:
        assert len(f["events/xs"]) == 0
        assert len(f["events/ts"]) == 0
        assert tuple(f["events"].attrs["sensor_shape"]) == (32, 32)


# ---- webcam command, fully stubbed ----------------------------------------------


class _FakeThreadedCapture:
    def __init__(self, source, capture_settings=None):
        self.frames = [
            np.full((24, 24, 3), 40 + i * 60, dtype=np.uint8) for i in range(4)
        ]

    def read(self, timeout=1.0):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        pass

    @property
    def actual_size(self):
        return (24, 24)


def test_webcam_command_runs_headless(monkeypatch):
    shown = []
    monkeypatch.setattr(cli_mod, "_ThreadedCapture", _FakeThreadedCapture)
    monkeypatch.setattr(cli_mod.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod.cv2, "imshow", lambda name, img: shown.append(img))
    monkeypatch.setattr(cli_mod.cv2, "waitKey", lambda ms: 0)
    monkeypatch.setattr(cli_mod.cv2, "destroyAllWindows", lambda: None)

    result = runner.invoke(app, ["webcam", "--threshold", "0.1"])

    assert result.exit_code == 0
    # 4 frames -> first primes prev_gray, 3 are rendered.
    assert len(shown) == 3
    for img in shown:
        assert img.shape == (24, 24, 3)
        assert img.dtype == np.uint8


def test_webcam_command_quits_on_q(monkeypatch):
    shown = []
    monkeypatch.setattr(cli_mod, "_ThreadedCapture", _FakeThreadedCapture)
    monkeypatch.setattr(cli_mod.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod.cv2, "imshow", lambda name, img: shown.append(img))
    monkeypatch.setattr(cli_mod.cv2, "waitKey", lambda ms: ord("q"))
    monkeypatch.setattr(cli_mod.cv2, "destroyAllWindows", lambda: None)

    result = runner.invoke(app, ["webcam"])

    assert result.exit_code == 0
    assert len(shown) == 1  # quit right after the first rendered frame
