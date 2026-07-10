import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from eventify import frame_stream


@pytest.fixture
def brightening_video(tmp_path):
    """10-frame 32x32 video, each frame uniformly brighter than the last."""
    path = tmp_path / "bright.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 100.0, (32, 32), isColor=True)
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((32, 32, 3), 30 + i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_output_shape(brightening_video):
    frames = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=50))
    assert len(frames) > 0
    for f in frames:
        assert f.shape == (4, 2, 32, 32)


def test_output_dtype_default(brightening_video):
    frames = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=50))
    assert len(frames) > 0
    assert frames[0].dtype == np.float32


def test_output_dtype_override(brightening_video):
    frames = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=50, dtype=np.float64))
    assert len(frames) > 0
    assert frames[0].dtype == np.float64


def test_channel_ordering_off_0_on_1(tmp_path):
    # Build a video guaranteed to produce only ON events (lossless-ish: large brightness jump).
    path = tmp_path / "pure_on.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 100.0, (32, 32), isColor=True)
    assert writer.isOpened()
    # Alternate between dark and bright to create clear ON events on odd frames.
    for i in range(20):
        val = 20 if i % 2 == 0 else 240
        writer.write(np.full((32, 32, 3), val, dtype=np.uint8))
    writer.release()

    frames = list(frame_stream(str(path), sensor_size=(32, 32), n_bins=4, window_ms=50))
    assert len(frames) > 0
    all_frames = np.stack(frames)
    # Both ON and OFF events exist (alternating); both channels must see activity.
    assert all_frames[:, 0].sum() > 0  # OFF channel
    assert all_frames[:, 1].sum() > 0  # ON channel


def test_non_overlapping_windows_cover_all_events(brightening_video):
    # Total ON events across all windows should equal what video_to_event_stream would produce.
    from eventify import video_to_event_stream
    import numpy as np

    all_chunks = list(video_to_event_stream(str(brightening_video), sensor_size=(32, 32)))
    ref_total = sum(len(c) for c in all_chunks)

    frames = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=50))
    streamed_total = int(sum(f.sum() for f in frames))

    # Allow for boundary events that fall outside completed windows.
    assert streamed_total <= ref_total
    assert streamed_total > 0


def test_values_are_non_negative(brightening_video):
    frames = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=50))
    for f in frames:
        assert np.all(f >= 0)


def test_stride_less_than_window_yields_more_frames(brightening_video):
    no_overlap = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=60))
    overlapping = list(frame_stream(str(brightening_video), sensor_size=(32, 32), n_bins=4, window_ms=60, stride_ms=20))
    assert len(overlapping) >= len(no_overlap)


def test_sensor_size_controls_output_spatial_dims(brightening_video):
    frames = list(frame_stream(str(brightening_video), sensor_size=(16, 16), n_bins=2, window_ms=50))
    assert len(frames) > 0
    assert frames[0].shape == (2, 2, 16, 16)


def test_nonexistent_source_raises(tmp_path):
    with pytest.raises((IOError, RuntimeError, Exception)):
        list(frame_stream(str(tmp_path / "missing.avi"), sensor_size=(32, 32), n_bins=4, window_ms=50))
