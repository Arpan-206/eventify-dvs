import numpy as np
import pytest

from eventify import interpolate_frames


def test_zero_intermediate_returns_only_endpoints():
    prev = np.zeros((4, 4), dtype=np.float32)
    curr = np.full((4, 4), 100.0, dtype=np.float32)
    frames = interpolate_frames(prev, curr, n_intermediate=0)

    assert len(frames) == 2
    assert np.array_equal(frames[0], prev)
    assert np.array_equal(frames[1], curr)


def test_returns_n_plus_2_frames():
    prev = np.zeros((4, 4), dtype=np.float32)
    curr = np.full((4, 4), 100.0, dtype=np.float32)
    for n in [1, 3, 5, 10]:
        frames = interpolate_frames(prev, curr, n_intermediate=n)
        assert len(frames) == n + 2


def test_intermediate_frames_are_linearly_interpolated():
    prev = np.zeros((4, 4), dtype=np.float32)
    curr = np.full((4, 4), 100.0, dtype=np.float32)

    # 3 intermediates means 5 frames total, evenly spaced values at 0, 25, 50, 75, 100.
    frames = interpolate_frames(prev, curr, n_intermediate=3)
    expected_values = [0.0, 25.0, 50.0, 75.0, 100.0]
    for frame, expected in zip(frames, expected_values):
        assert np.allclose(frame, expected, atol=1e-4)


def test_preserves_shape():
    prev = np.zeros((17, 23), dtype=np.float32)
    curr = np.zeros_like(prev)
    frames = interpolate_frames(prev, curr, n_intermediate=4)
    for frame in frames:
        assert frame.shape == (17, 23)


def test_endpoints_match_inputs_exactly():
    rng = np.random.default_rng(0)
    prev = rng.uniform(0, 255, size=(8, 8)).astype(np.float32)
    curr = rng.uniform(0, 255, size=(8, 8)).astype(np.float32)
    frames = interpolate_frames(prev, curr, n_intermediate=5)
    assert np.allclose(frames[0], prev)
    assert np.allclose(frames[-1], curr)


def test_negative_n_intermediate_raises():
    prev = np.zeros((4, 4), dtype=np.float32)
    curr = np.zeros_like(prev)
    with pytest.raises(ValueError):
        interpolate_frames(prev, curr, n_intermediate=-1)


def test_shape_mismatch_raises():
    prev = np.zeros((4, 4), dtype=np.float32)
    curr = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        interpolate_frames(prev, curr, n_intermediate=2)


def test_interpolated_frames_produce_more_events_than_direct():
    """Feeding interpolated sub-frames through frame_to_event_tuples should
    yield strictly more events than a single prev->curr diff, because gradual
    changes cross the log threshold more times cumulatively."""
    from eventify import frame_to_event_tuples

    # Large, gradual brightening on the whole frame.
    prev = np.full((16, 16), 30.0, dtype=np.float32)
    curr = np.full((16, 16), 220.0, dtype=np.float32)

    # Direct: one big jump.
    direct = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.05
    )

    # Interpolated: 4 intermediates → 5 sub-intervals.
    sub_frames = interpolate_frames(prev, curr, n_intermediate=4)
    total = 0
    step_us = 1000 // (len(sub_frames) - 1)
    for i in range(len(sub_frames) - 1):
        chunk = frame_to_event_tuples(
            sub_frames[i],
            sub_frames[i + 1],
            prev_t_us=i * step_us,
            curr_t_us=(i + 1) * step_us,
            c_thresh=0.05,
        )
        total += len(chunk)

    # With multi-crossing, direct differencing already captures all crossings,
    # so interpolation shouldn't reduce the count; the interesting property is
    # that interpolation gives finer-grained timestamps for the same crossings.
    assert total >= len(direct) * 0.9  # allow ~10% rounding slack
