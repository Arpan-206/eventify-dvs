import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")

from eventify.cli import _events_to_frame, _parse_sensor_size, main


# ---- _parse_sensor_size -------------------------------------------------------


def test_parse_sensor_size_valid():
    assert _parse_sensor_size("128,64") == (128, 64)


def test_parse_sensor_size_missing_comma():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_sensor_size("128x64")


def test_parse_sensor_size_non_numeric():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_sensor_size("w,h")


def test_parse_sensor_size_extra_parts():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_sensor_size("128,64,32")


# ---- _events_to_frame ---------------------------------------------------------


def test_events_to_frame_empty_chunk_returns_black():
    from eventify import EVENT_DTYPE
    chunk = np.zeros(0, dtype=EVENT_DTYPE)
    img = _events_to_frame(chunk, h=16, w=16)
    assert img.shape == (16, 16, 3)
    assert img.dtype == np.uint8
    assert img.sum() == 0


def test_events_to_frame_single_on_event_renders_blue():
    from eventify import EVENT_DTYPE
    chunk = np.zeros(1, dtype=EVENT_DTYPE)
    chunk["x"] = 5
    chunk["y"] = 3
    chunk["p"] = 1
    img = _events_to_frame(chunk, h=16, w=16)
    # ON pixel: blue channel dominant (BGR: 180, 70, 0).
    px = img[3, 5]
    assert px[0] > px[2]  # B > R
    assert img[0, 0].sum() == 0  # background stays black


def test_events_to_frame_single_off_event_renders_amber():
    from eventify import EVENT_DTYPE
    chunk = np.zeros(1, dtype=EVENT_DTYPE)
    chunk["x"] = 2
    chunk["y"] = 7
    chunk["p"] = 0
    img = _events_to_frame(chunk, h=16, w=16)
    # OFF pixel: red+green dominant (BGR: 0, 170, 220).
    px = img[7, 2]
    assert px[2] > px[0]  # R > B
    assert img[0, 0].sum() == 0


def test_events_to_frame_output_shape():
    from eventify import EVENT_DTYPE
    chunk = np.zeros(0, dtype=EVENT_DTYPE)
    img = _events_to_frame(chunk, h=24, w=32)
    assert img.shape == (24, 32, 3)


# ---- CLI end-to-end -----------------------------------------------------------


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "input.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for i in range(5):
        val = 40 + i * 40
        writer.write(np.full((32, 32, 3), val, dtype=np.uint8))
    writer.release()
    return path


def test_convert_produces_output_video(synthetic_video, tmp_path):
    out = tmp_path / "out.mp4"
    result = main(["convert", str(synthetic_video), str(out)])
    assert result == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_convert_output_has_correct_frame_count(synthetic_video, tmp_path):
    out = tmp_path / "out.mp4"
    main(["convert", str(synthetic_video), str(out)])
    cap = cv2.VideoCapture(str(out))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 5 input frames → 4 event frames.
    assert count == 4


def test_convert_nonexistent_input_returns_error(tmp_path):
    out = tmp_path / "out.mp4"
    result = main(["convert", str(tmp_path / "missing.avi"), str(out)])
    assert result != 0


def test_export_produces_hdf5(synthetic_video, tmp_path):
    out = tmp_path / "events.h5"
    result = main(["export", str(synthetic_video), str(out)])
    assert result == 0
    assert out.exists()
    with h5py.File(str(out), "r") as f:
        assert "events" in f
        assert "xs" in f["events"]


def test_export_sensor_size_override(synthetic_video, tmp_path):
    out = tmp_path / "events.h5"
    main(["export", str(synthetic_video), str(out), "--sensor-size", "16,16"])
    with h5py.File(str(out), "r") as f:
        assert tuple(f["events"].attrs["sensor_shape"]) == (16, 16)
        assert f["events/xs"][:].max() < 16
        assert f["events/ys"][:].max() < 16


def test_export_interp_produces_events(synthetic_video, tmp_path):
    # Verify the interp code path runs and produces a valid non-empty HDF5.
    # Interpolation splits each frame-pair into sub-intervals, so each sub-step
    # sees a smaller log-delta and may cross fewer thresholds — total event count
    # can be lower than the direct path, which is expected behaviour.
    out = tmp_path / "interp.h5"
    result = main(["export", str(synthetic_video), str(out), "--interp", "3"])
    assert result == 0
    with h5py.File(str(out), "r") as f:
        assert len(f["events/xs"]) > 0


def test_export_nonexistent_input_raises(tmp_path):
    out = tmp_path / "events.h5"
    with pytest.raises((IOError, SystemExit, Exception)):
        main(["export", str(tmp_path / "missing.avi"), str(out)])
