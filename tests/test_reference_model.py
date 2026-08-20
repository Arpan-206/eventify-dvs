"""Cross-check frame_to_event_tuples against a brute-force per-pixel reference.

The library builds its event list with vectorized repeat/cumsum machinery;
these tests re-derive the same events with plain Python loops and require the
two to agree exactly, then pin down behaviors no other test file covers:
eps handling, uint8 inputs, determinism, non-square sensor orientation, and
timestamp range guarantees.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from eventify import EVENT_DTYPE, frame_to_event_tuples


def _crossings_like_impl(prev, curr, c_thresh, eps):
    """Compute per-pixel crossing counts/polarity with the impl's exact dtypes."""
    prev_f = prev.astype(np.float32)
    curr_f = curr.astype(np.float32)
    delta = np.log(curr_f + eps) - np.log(prev_f + eps)
    crossings = np.floor(np.abs(delta) / c_thresh).astype(np.int32)
    return delta, crossings


def _reference_events(prev, curr, prev_t_us, curr_t_us, c_thresh, eps):
    """Brute-force expansion: one loop per pixel, one loop per crossing."""
    delta, crossings = _crossings_like_impl(prev, curr, c_thresh, eps)
    interval = curr_t_us - prev_t_us
    rows = []
    h, w = crossings.shape
    for y in range(h):
        for x in range(w):
            k_total = int(crossings[y, x])
            if k_total == 0:
                continue
            p = 1 if delta[y, x] > 0 else 0
            for k in range(1, k_total + 1):
                t = int(prev_t_us + (k / (k_total + 1)) * interval)
                rows.append((x, y, t, p))
    out = np.array(rows, dtype=EVENT_DTYPE) if rows else np.zeros(0, dtype=EVENT_DTYPE)
    return out


def _sorted(events):
    return np.sort(events, order=["y", "x", "t", "p"])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("c_thresh", [0.05, 0.15, 0.5])
def test_matches_bruteforce_expansion(seed, c_thresh):
    rng = np.random.default_rng(seed)
    prev = rng.uniform(0, 255, size=(17, 23)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 80, size=prev.shape), 0, 255).astype(np.float32)

    got = frame_to_event_tuples(
        prev, curr, prev_t_us=5_000, curr_t_us=1_005_000, c_thresh=c_thresh
    )
    want = _reference_events(
        prev, curr, prev_t_us=5_000, curr_t_us=1_005_000, c_thresh=c_thresh, eps=1.0
    )

    assert len(got) == len(want)
    assert np.array_equal(_sorted(got), _sorted(want))


def test_eps_forwarded_and_compresses_dark_response():
    # Same absolute change on a dark pixel: log((10+eps)/(1+eps)) shrinks as
    # eps grows, so a large eps must produce strictly fewer crossings.
    prev = np.full((4, 4), 1.0, dtype=np.float32)
    curr = np.full((4, 4), 10.0, dtype=np.float32)

    many = frame_to_event_tuples(prev, curr, 0, 1000, c_thresh=0.05, eps=1.0)
    few = frame_to_event_tuples(prev, curr, 0, 1000, c_thresh=0.05, eps=100.0)

    assert len(many) > len(few)
    # And the counts match the closed-form crossing formula.
    k_many = int(np.floor(np.log(11.0 / 2.0) / 0.05))
    assert len(many) == 16 * k_many


def test_uint8_input_matches_float32_input():
    rng = np.random.default_rng(3)
    prev8 = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    curr8 = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)

    from_u8 = frame_to_event_tuples(prev8, curr8, 0, 33_333, c_thresh=0.15)
    from_f32 = frame_to_event_tuples(
        prev8.astype(np.float32), curr8.astype(np.float32), 0, 33_333, c_thresh=0.15
    )
    assert np.array_equal(from_u8, from_f32)


def test_pure_function_is_deterministic():
    rng = np.random.default_rng(4)
    prev = rng.uniform(0, 255, size=(16, 16)).astype(np.float32)
    curr = rng.uniform(0, 255, size=(16, 16)).astype(np.float32)

    a = frame_to_event_tuples(prev, curr, 0, 1000)
    b = frame_to_event_tuples(prev, curr, 0, 1000)
    assert np.array_equal(a, b)


def test_nonsquare_sensor_size_is_width_height():
    # sensor_size is documented as (W, H): a (64, 32) request must yield
    # x in [0, 64) and y in [0, 32) — a swap would put x past 32.
    prev = np.full((100, 200), 50.0, dtype=np.float32)
    curr = np.full((100, 200), 250.0, dtype=np.float32)  # every pixel fires

    events = frame_to_event_tuples(prev, curr, 0, 1000, sensor_size=(64, 32))

    assert events["x"].max() == 63
    assert events["y"].max() == 31


def test_large_time_offsets_do_not_overflow():
    base = 10**15  # ~31 years in microseconds; must survive in int64
    prev = np.full((8, 8), 50.0, dtype=np.float32)
    curr = np.full((8, 8), 250.0, dtype=np.float32)

    events = frame_to_event_tuples(prev, curr, base, base + 1_000_000)

    assert events["t"].dtype == np.int64
    assert np.all(events["t"] >= base)
    assert np.all(events["t"] < base + 1_000_000)


def test_timestamps_never_reach_interval_end():
    # Stagger alphas are k/(K+1) < 1, so even after int truncation every
    # timestamp must land in [prev_t_us, curr_t_us).
    rng = np.random.default_rng(5)
    prev = rng.uniform(0, 255, size=(16, 16)).astype(np.float32)
    curr = rng.uniform(0, 255, size=(16, 16)).astype(np.float32)

    events = frame_to_event_tuples(prev, curr, 200, 1200)
    assert np.all(events["t"] >= 200)
    assert np.all(events["t"] < 1200)


def test_per_pixel_timestamps_strictly_increase_when_interval_is_wide():
    prev = np.full((4, 4), 20.0, dtype=np.float32)
    curr = np.full((4, 4), 250.0, dtype=np.float32)  # many crossings per pixel

    events = frame_to_event_tuples(prev, curr, 0, 1_000_000)

    for y in range(4):
        for x in range(4):
            ts = events["t"][(events["x"] == x) & (events["y"] == y)]
            assert len(ts) > 1
            assert np.all(np.diff(ts) > 0)


def test_nonpositive_c_thresh_raises():
    prev = np.full((4, 4), 100.0, dtype=np.float32)
    with pytest.raises(ValueError):
        frame_to_event_tuples(prev, prev.copy(), 0, 1000, c_thresh=0.0)
    with pytest.raises(ValueError):
        frame_to_event_tuples(prev, prev.copy(), 0, 1000, c_thresh=-0.1)


def test_nonpositive_eps_raises():
    prev = np.full((4, 4), 100.0, dtype=np.float32)
    with pytest.raises(ValueError):
        frame_to_event_tuples(prev, prev.copy(), 0, 1000, eps=0.0)
