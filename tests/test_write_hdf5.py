import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from eventify import EVENT_DTYPE, write_hdf5


def _make_events(n=10):
    ev = np.zeros(n, dtype=EVENT_DTYPE)
    ev["x"] = np.arange(n, dtype=np.int16)
    ev["y"] = np.arange(n, dtype=np.int16) * 2
    ev["t"] = np.arange(n, dtype=np.int64) * 100
    ev["p"] = np.tile([0, 1], (n + 1) // 2)[:n]
    return ev


def test_writes_dvs_gesture_style_layout(tmp_path):
    path = tmp_path / "events.h5"
    events = _make_events(20)
    write_hdf5(str(path), events, sensor_shape=(128, 128))

    with h5py.File(str(path), "r") as f:
        assert "events" in f
        grp = f["events"]
        # DVS-Gesture reprocessed convention: xs / ys / ts / ps datasets.
        for name in ("xs", "ys", "ts", "ps"):
            assert name in grp, f"missing dataset events/{name}"


def test_dataset_values_roundtrip(tmp_path):
    path = tmp_path / "events.h5"
    events = _make_events(20)
    write_hdf5(str(path), events, sensor_shape=(128, 128))

    with h5py.File(str(path), "r") as f:
        assert np.array_equal(f["events/xs"][:], events["x"])
        assert np.array_equal(f["events/ys"][:], events["y"])
        assert np.array_equal(f["events/ts"][:], events["t"])
        assert np.array_equal(f["events/ps"][:], events["p"])


def test_sensor_shape_stored_as_attr(tmp_path):
    path = tmp_path / "events.h5"
    write_hdf5(str(path), _make_events(5), sensor_shape=(128, 128))

    with h5py.File(str(path), "r") as f:
        shape = tuple(f["events"].attrs["sensor_shape"])
        assert shape == (128, 128)


def test_polarity_dataset_is_binary(tmp_path):
    path = tmp_path / "events.h5"
    write_hdf5(str(path), _make_events(50), sensor_shape=(128, 128))
    with h5py.File(str(path), "r") as f:
        ps = f["events/ps"][:]
        assert set(np.unique(ps).tolist()).issubset({0, 1})


def test_empty_events_writes_valid_file(tmp_path):
    path = tmp_path / "empty.h5"
    empty = np.zeros(0, dtype=EVENT_DTYPE)
    write_hdf5(str(path), empty, sensor_shape=(128, 128))

    with h5py.File(str(path), "r") as f:
        assert f["events/xs"].shape == (0,)
        assert f["events/ts"].shape == (0,)


def test_accepts_pathlib_path(tmp_path):
    from pathlib import Path

    path = Path(tmp_path) / "events.h5"
    write_hdf5(path, _make_events(3), sensor_shape=(64, 64))
    assert path.exists()


def test_dataset_dtypes_match_spec(tmp_path):
    path = tmp_path / "events.h5"
    write_hdf5(str(path), _make_events(10), sensor_shape=(128, 128))
    with h5py.File(str(path), "r") as f:
        assert f["events/xs"].dtype == np.dtype("<i2")
        assert f["events/ys"].dtype == np.dtype("<i2")
        assert f["events/ts"].dtype == np.dtype("<i8")
        assert f["events/ps"].dtype == np.dtype("i1")
