"""Fast primitives for the low-latency webcam preview."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_log_lut(eps: float = 1.0) -> np.ndarray:
    """Precompute ``log(x + eps)`` for x = 0..255 as a float32 array."""
    x = np.arange(256, dtype=np.float32)
    return np.log(x + np.float32(eps))


def frame_to_crossing_counts(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    c_thresh: float = 0.05,
    log_lut: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(on_counts, off_counts)`` 2D int arrays. Inputs must be uint8 grayscale."""
    if prev_frame.dtype != np.uint8 or curr_frame.dtype != np.uint8:
        raise TypeError(
            "frame_to_crossing_counts requires uint8 inputs; "
            f"got {prev_frame.dtype} and {curr_frame.dtype}"
        )
    if prev_frame.shape != curr_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )

    if log_lut is None:
        log_lut = build_log_lut(eps=1.0)

    delta = log_lut[curr_frame] - log_lut[prev_frame]
    crossings = np.floor_divide(np.abs(delta), c_thresh).astype(np.int16)
    positive = delta > 0
    return np.where(positive, crossings, 0), np.where(positive, 0, crossings)
