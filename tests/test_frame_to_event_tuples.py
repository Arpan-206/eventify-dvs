import numpy as np
import pytest

from eventify import EVENT_DTYPE, frame_to_event_tuples


def _uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


def test_returns_structured_event_array():
    prev = _uniform((8, 8), 100.0)
    curr = prev.copy()
    curr[2:4, 3:5] = 250.0
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert isinstance(events, np.ndarray)
    assert events.dtype == EVENT_DTYPE
    # Fields present and named correctly.
    assert set(events.dtype.names) == {"x", "y", "t", "p"}


def test_identical_frames_produce_no_events():
    prev = _uniform((8, 8), 100.0)
    events = frame_to_event_tuples(prev, prev.copy(), prev_t_us=0, curr_t_us=1000)
    assert len(events) == 0


def test_brightening_pixel_barely_crossing_yields_one_event():
    # log(120/100) ≈ 0.182 → at c_thresh=0.15, exactly 1 crossing.
    prev = _uniform((4, 4), 99.0)  # +eps=1 → 100 in log
    curr = prev.copy()
    curr[1, 2] = 119.0  # +eps=1 → 120
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )

    assert len(events) == 1
    e = events[0]
    assert (e["x"], e["y"]) == (2, 1)
    assert e["p"] == 1


def test_darkening_pixel_barely_crossing_yields_one_event():
    prev = _uniform((4, 4), 119.0)
    curr = prev.copy()
    curr[3, 0] = 99.0
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )

    assert len(events) == 1
    e = events[0]
    assert (e["x"], e["y"]) == (0, 3)
    assert e["p"] == 0


def test_polarity_is_binary_only():
    rng = np.random.default_rng(0)
    prev = rng.uniform(20, 220, size=(32, 32)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 50, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert set(np.unique(events["p"]).tolist()).issubset({0, 1})


def test_timestamps_lie_in_interval():
    rng = np.random.default_rng(0)
    prev = rng.uniform(20, 220, size=(16, 16)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 60, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=1000, curr_t_us=2000)

    assert len(events) > 0
    assert np.all(events["t"] >= 1000)
    assert np.all(events["t"] <= 2000)


def test_timestamps_uniformly_distributed():
    # A large event count should span most of the interval, not cluster at one end.
    rng = np.random.default_rng(1)
    prev = rng.uniform(20, 220, size=(64, 64)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 80, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=10000)

    ts = events["t"]
    assert len(ts) >= 100
    # Spread should cover a large fraction of the interval.
    assert ts.min() < 2000
    assert ts.max() > 8000


def test_coordinates_within_frame_bounds():
    rng = np.random.default_rng(2)
    prev = rng.uniform(20, 220, size=(16, 24)).astype(np.float32)  # h=16, w=24
    curr = np.clip(prev + rng.normal(0, 60, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert np.all(events["x"] >= 0) and np.all(events["x"] < 24)
    assert np.all(events["y"] >= 0) and np.all(events["y"] < 16)


def test_sensor_size_resizes_before_event_gen():
    prev = _uniform((100, 200), 100.0)
    curr = prev.copy()
    curr[:] = 250.0  # global brightening -> every pixel fires

    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, sensor_size=(32, 32)  # (w, h)
    )
    # After resize to 32x32; with multi-crossing, we can get up to (32*32)*K events
    # where K is the per-pixel crossing count. Just check coord bounds hold.
    assert np.all(events["x"] < 32)
    assert np.all(events["y"] < 32)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        frame_to_event_tuples(
            _uniform((4, 4), 100.0),
            _uniform((5, 5), 100.0),
            prev_t_us=0,
            curr_t_us=1000,
        )


def test_negative_interval_raises():
    prev = _uniform((4, 4), 100.0)
    curr = _uniform((4, 4), 200.0)
    with pytest.raises(ValueError):
        frame_to_event_tuples(prev, curr, prev_t_us=1000, curr_t_us=500)


# ---- multi-crossing tests -------------------------------------------------


def test_multi_crossing_emits_multiple_events_per_pixel():
    """A pixel whose log-delta spans K thresholds should emit K events."""
    # log(200/50) ≈ 1.386. At c_thresh=0.15, floor(1.386/0.15) = 9 crossings.
    prev = _uniform((3, 3), 49.0)  # +eps=1 → 50
    curr = prev.copy()
    curr[1, 1] = 199.0  # +eps=1 → 200
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )

    # Exactly one pixel fired, but many crossings.
    assert len(events) == 9
    # All events for that single pixel share coordinates and polarity.
    assert np.all(events["x"] == 1)
    assert np.all(events["y"] == 1)
    assert np.all(events["p"] == 1)


def test_multi_crossing_timestamps_are_distinct_and_ordered():
    prev = _uniform((1, 1), 49.0)
    curr = _uniform((1, 1), 199.0)
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=9000, c_thresh=0.15
    )

    ts = events["t"]
    assert len(ts) == 9
    # Strictly monotonic within a single pixel's event train.
    assert np.all(np.diff(ts) > 0)
    # Spread across the interval, not clustered at either end.
    assert ts.min() >= 0
    assert ts.max() <= 9000


def test_multi_crossing_darkening_emits_multiple_off_events():
    prev = _uniform((2, 2), 199.0)
    curr = prev.copy()
    curr[0, 0] = 49.0
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )

    assert len(events) == 9
    assert np.all(events["p"] == 0)


def test_sub_threshold_change_still_yields_no_events():
    prev = _uniform((4, 4), 100.0)
    curr = _uniform((4, 4), 105.0)  # log(106/101) ≈ 0.048, below 0.15
    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )
    assert len(events) == 0


def test_default_threshold_is_lower_than_before():
    """Default c_thresh should be 0.05 for more sensitivity."""
    import inspect

    sig = inspect.signature(frame_to_event_tuples)
    assert sig.parameters["c_thresh"].default == 0.05


def test_lower_default_produces_more_events_than_old_default():
    """Same input, default threshold now vs. old 0.15 → strictly more events."""
    rng = np.random.default_rng(42)
    prev = rng.uniform(20, 220, size=(32, 32)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 15, size=prev.shape), 1, 255).astype(np.float32)

    events_default = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)
    events_old = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.15
    )
    assert len(events_default) > len(events_old)


def test_bgr_input_is_converted_to_grayscale():
    # A BGR frame that differs from prev only in the blue channel should still fire.
    prev = np.full((8, 8, 3), 50, dtype=np.uint8)
    curr = np.full((8, 8, 3), 50, dtype=np.uint8)
    curr[:, :, 0] = 200  # boost blue channel
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)
    # Grayscale of prev ≈ 50, grayscale of curr ≈ 86 — should cross threshold.
    assert len(events) > 0


def test_zero_interval_events_all_at_boundary():
    # When prev_t_us == curr_t_us every timestamp must equal that value.
    prev = np.full((4, 4), 50.0, dtype=np.float32)
    curr = np.full((4, 4), 200.0, dtype=np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=5000, curr_t_us=5000)
    assert len(events) > 0
    assert np.all(events["t"] == 5000)
