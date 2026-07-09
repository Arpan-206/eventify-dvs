"""Tests for the fast rendering path: LUT + integer crossing-count maps.

These primitives skip per-event timestamp assignment and stay in vectorized
NumPy from start to finish, for use in the low-latency webcam preview.
"""

import numpy as np
import pytest

from eventify.fast import build_log_lut, frame_to_crossing_counts


def test_log_lut_matches_np_log():
    """The LUT must reproduce log(x + eps) to full float32 precision for uint8 in."""
    lut = build_log_lut(eps=1.0)
    assert lut.shape == (256,)
    assert lut.dtype == np.float32

    x = np.arange(256, dtype=np.uint8)
    expected = np.log(x.astype(np.float32) + 1.0)
    assert np.allclose(lut, expected, atol=1e-5)


def test_log_lut_honors_eps():
    lut_1 = build_log_lut(eps=1.0)
    lut_5 = build_log_lut(eps=5.0)
    assert not np.allclose(lut_1, lut_5)
    assert np.isclose(lut_5[0], np.log(5.0), atol=1e-5)


def test_frame_to_crossing_counts_shape_and_dtype():
    prev = np.zeros((32, 32), dtype=np.uint8)
    curr = np.full((32, 32), 200, dtype=np.uint8)
    on, off = frame_to_crossing_counts(prev, curr, c_thresh=0.15)

    assert on.shape == (32, 32)
    assert off.shape == (32, 32)
    # Small ints — sensor never crosses more than ~log(256/1)/0.05 ≈ 110 times.
    assert on.dtype.kind in ("i", "u")
    assert off.dtype.kind in ("i", "u")


def test_identical_frames_produce_no_crossings():
    prev = np.full((16, 16), 100, dtype=np.uint8)
    on, off = frame_to_crossing_counts(prev, prev.copy(), c_thresh=0.05)
    assert on.sum() == 0
    assert off.sum() == 0


def test_brightening_produces_on_crossings_only():
    prev = np.full((8, 8), 50, dtype=np.uint8)
    curr = np.full((8, 8), 200, dtype=np.uint8)
    on, off = frame_to_crossing_counts(prev, curr, c_thresh=0.15)

    # log(201/51) ≈ 1.37 → floor(1.37/0.15) = 9 crossings per pixel.
    assert np.all(on == 9)
    assert np.all(off == 0)


def test_darkening_produces_off_crossings_only():
    prev = np.full((8, 8), 200, dtype=np.uint8)
    curr = np.full((8, 8), 50, dtype=np.uint8)
    on, off = frame_to_crossing_counts(prev, curr, c_thresh=0.15)
    assert np.all(off == 9)
    assert np.all(on == 0)


def test_agrees_with_reference_frame_to_event_tuples():
    """The fast counts must match the total-per-pixel event count from the
    reference implementation, so downstream numbers stay consistent."""
    from eventify import frame_to_event_tuples

    rng = np.random.default_rng(42)
    prev = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
    curr = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)

    on, off = frame_to_crossing_counts(prev, curr, c_thresh=0.1)
    ref = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000, c_thresh=0.1)

    # Reference emits (on.sum() + off.sum()) individual events.
    assert on.sum() + off.sum() == len(ref)


def test_accepts_precomputed_lut():
    """Callers who want to reuse the LUT across many calls can pass it in."""
    lut = build_log_lut(eps=1.0)
    prev = np.full((4, 4), 50, dtype=np.uint8)
    curr = np.full((4, 4), 200, dtype=np.uint8)

    on_a, off_a = frame_to_crossing_counts(prev, curr, c_thresh=0.15)
    on_b, off_b = frame_to_crossing_counts(prev, curr, c_thresh=0.15, log_lut=lut)
    assert np.array_equal(on_a, on_b)
    assert np.array_equal(off_a, off_b)


def test_rejects_non_uint8():
    prev = np.full((4, 4), 50, dtype=np.float32)
    curr = np.full((4, 4), 200, dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        frame_to_crossing_counts(prev, curr, c_thresh=0.15)
