import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from eventify import EVENT_DTYPE, video_to_event_stream


@pytest.fixture
def synthetic_video(tmp_path):
    """5-frame 32x32 video, each frame uniformly brighter than the last."""
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for i in range(5):
        val = 40 + i * 40
        writer.write(np.full((32, 32, 3), val, dtype=np.uint8))
    writer.release()
    return path


def test_yields_structured_event_chunks(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video)))
    assert len(chunks) == 4  # N-1 chunks for N=5 frames
    for chunk in chunks:
        assert chunk.dtype == EVENT_DTYPE


def test_all_polarities_are_one_for_brightening_video(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video), c_thresh=0.05))
    all_events = np.concatenate(chunks)
    assert len(all_events) > 0
    assert np.all(all_events["p"] == 1)


def test_timestamps_are_monotonic_across_chunks(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video)))
    last_max = -1
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        assert chunk["t"].min() >= last_max
        last_max = int(chunk["t"].max())


def test_timestamps_are_microseconds(synthetic_video):
    # At 10 FPS the frame period is 100_000 µs. Events across the whole
    # 5-frame clip should span roughly that many µs per inter-frame gap.
    chunks = list(video_to_event_stream(str(synthetic_video)))
    all_events = np.concatenate(chunks)
    # Total span >> milliseconds -> clearly microseconds, not seconds.
    assert all_events["t"].max() > 100_000


def test_sensor_size_override_reshapes_events(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video), sensor_size=(16, 16)))
    all_events = np.concatenate(chunks)
    assert np.all(all_events["x"] < 16)
    assert np.all(all_events["y"] < 16)


def test_native_resolution_when_sensor_size_omitted(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video)))
    all_events = np.concatenate(chunks)
    # Source is 32x32; without override, coords must fit that.
    assert all_events["x"].max() < 32
    assert all_events["y"].max() < 32


def test_end_of_stream_graceful(synthetic_video):
    gen = video_to_event_stream(str(synthetic_video))
    for _ in gen:
        pass
    assert list(gen) == []


def test_nonexistent_source_raises(tmp_path):
    missing = tmp_path / "does_not_exist.avi"
    with pytest.raises((IOError, RuntimeError, ValueError)):
        list(video_to_event_stream(str(missing)))


@pytest.fixture
def static_video(tmp_path):
    """5-frame 32x32 video where every frame is identical."""
    path = tmp_path / "static.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((32, 32, 3), 128, dtype=np.uint8))
    writer.release()
    return path


def test_identical_frames_produce_zero_events(static_video):
    chunks = list(video_to_event_stream(str(static_video)))
    total = sum(len(c) for c in chunks)
    assert total == 0


def test_interp_path_yields_more_chunks(synthetic_video):
    # interp=3 means 4 sub-intervals per frame pair, so 4x as many chunks.
    chunks_no_interp = list(video_to_event_stream(str(synthetic_video), interp=0))
    chunks_interp = list(video_to_event_stream(str(synthetic_video), interp=3))
    assert len(chunks_interp) == len(chunks_no_interp) * 4


def test_interp_path_timestamps_are_monotonic(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video), interp=2))
    last_max = -1
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        assert chunk["t"].min() >= last_max
        last_max = int(chunk["t"].max())


def test_interp_path_all_events_have_binary_polarity(synthetic_video):
    chunks = list(video_to_event_stream(str(synthetic_video), interp=2))
    all_events = np.concatenate(chunks)
    assert len(all_events) > 0
    assert set(np.unique(all_events["p"]).tolist()).issubset({0, 1})
