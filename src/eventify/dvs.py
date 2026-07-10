"""DVS-style event simulation: binary-polarity (x, y, t_us, p) tuples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import h5py
import numpy as np

# NumPy structured dtype for a single DVS event.
# int16 coords cover any realistic sensor; int64 for µs timestamps to
# hold long captures without overflow. Polarity ∈ {0, 1}.
EVENT_DTYPE = np.dtype([("x", "<i2"), ("y", "<i2"), ("t", "<i8"), ("p", "<i1")])


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32, copy=False)


def _maybe_resize(
    frame: np.ndarray, sensor_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    if sensor_size is None:
        return frame
    w, h = sensor_size
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def interpolate_frames(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    n_intermediate: int,
) -> list:
    """Linearly interpolate ``n_intermediate`` frames between two endpoints.

    Returns a list of length ``n_intermediate + 2`` starting with ``prev_frame``
    and ending with ``curr_frame``. A naive stand-in for optical-flow-based
    interpolation (v2e's SuperSloMo) that keeps the library dependency-free.
    """
    if prev_frame.shape != curr_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )
    if n_intermediate < 0:
        raise ValueError(f"n_intermediate must be >= 0, got {n_intermediate}")

    prev_f = prev_frame.astype(np.float32, copy=False)
    curr_f = curr_frame.astype(np.float32, copy=False)

    total = n_intermediate + 2
    alphas = np.linspace(0.0, 1.0, total)
    return [(1.0 - a) * prev_f + a * curr_f for a in alphas]


def frame_to_event_tuples(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_t_us: int,
    curr_t_us: int,
    c_thresh: float = 0.05,
    eps: float = 1.0,
    sensor_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Emit binary-polarity DVS event tuples with multi-crossing.

    A pixel whose log-intensity change spans ``K`` threshold widths emits
    ``K`` events (matching how a real DVS sensor fires once per crossing).
    Each pixel's events are staggered uniformly across
    ``[prev_t_us, curr_t_us]`` so their timestamps are strictly monotonic.
    """
    if prev_frame.shape != curr_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )
    if curr_t_us < prev_t_us:
        raise ValueError(f"curr_t_us ({curr_t_us}) must be >= prev_t_us ({prev_t_us})")

    prev = _maybe_resize(_to_gray_float(prev_frame), sensor_size)
    curr = _maybe_resize(_to_gray_float(curr_frame), sensor_size)

    delta = np.log(curr + eps) - np.log(prev + eps)

    # How many full c_thresh crossings did each pixel span?
    crossings = np.floor(np.abs(delta) / c_thresh).astype(np.int32)
    total_events = int(crossings.sum())
    if total_events == 0:
        return np.zeros(0, dtype=EVENT_DTYPE)

    # Flat arrays of (x, y, polarity) counts, one entry per firing pixel.
    ys, xs = np.nonzero(crossings)
    counts = crossings[ys, xs]
    polarities = (delta[ys, xs] > 0).astype(np.int8)  # 1 = ON, 0 = OFF

    # Expand: each pixel contributes `counts[i]` copies of its coords/polarity.
    events = np.zeros(total_events, dtype=EVENT_DTYPE)
    events["x"] = np.repeat(xs.astype(np.int16), counts)
    events["y"] = np.repeat(ys.astype(np.int16), counts)
    events["p"] = np.repeat(polarities, counts)

    # Timestamps: for each pixel with K events, stagger them uniformly across
    # the frame interval. Position k of K → alpha = (k+1)/(K+1) so no event
    # lands exactly on the boundary, and consecutive events at the same pixel
    # are strictly ordered.
    #
    # Vectorized construction of per-pixel indices [1..K1, 1..K2, ...]:
    # start_indices marks where each pixel's block begins in the flat array;
    # a running cumsum minus the offset per block yields 1..K per pixel.
    interval = curr_t_us - prev_t_us
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    within_pixel_idx = (np.arange(total_events) - np.repeat(starts, counts) + 1).astype(
        np.float64
    )
    within_pixel_denom = np.repeat(counts + 1, counts).astype(np.float64)
    alphas = within_pixel_idx / within_pixel_denom
    events["t"] = (prev_t_us + alphas * interval).astype(np.int64)

    return events


def video_to_event_stream(
    source: Union[str, int],
    c_thresh: float = 0.05,
    sensor_size: Optional[Tuple[int, int]] = None,
    interp: int = 0,
    capture_settings: Optional[dict] = None,
) -> Generator[np.ndarray, None, None]:
    """Yield per-frame-pair structured event arrays from a video or webcam.

    ``source`` is a file path or an integer webcam device index. When
    ``interp > 0``, that many sub-frames are linearly interpolated between
    each real frame pair and event generation runs on every sub-interval,
    yielding one chunk per sub-interval. ``capture_settings`` is an optional
    dict of OpenCV ``CAP_PROP_*`` overrides (e.g. width/height/fps) applied
    right after the capture opens.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video source: {source!r}")

    if capture_settings:
        for prop, value in capture_settings.items():
            cap.set(prop, value)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_period_us = (
        int(1_000_000 / fps) if fps > 0 else 33_333
    )  # ~30 FPS webcam fallback

    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            return
        prev_t_us = 0

        prev_processed = _maybe_resize(_to_gray_float(prev), sensor_size)

        frame_idx = 1
        while True:
            ok, curr = cap.read()
            if not ok or curr is None:
                break

            curr_processed = _maybe_resize(_to_gray_float(curr), sensor_size)

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                curr_t_us = int(pos_ms * 1000)
            else:
                curr_t_us = frame_idx * frame_period_us

            if interp <= 0:
                yield frame_to_event_tuples(
                    prev_processed,
                    curr_processed,
                    prev_t_us=prev_t_us,
                    curr_t_us=curr_t_us,
                    c_thresh=c_thresh,
                )
            else:
                sub_frames = interpolate_frames(
                    prev_processed, curr_processed, n_intermediate=interp
                )
                n_sub = len(sub_frames) - 1  # number of sub-intervals
                span = curr_t_us - prev_t_us
                for i in range(n_sub):
                    sub_prev_t = prev_t_us + i * span // n_sub
                    sub_curr_t = prev_t_us + (i + 1) * span // n_sub
                    yield frame_to_event_tuples(
                        sub_frames[i],
                        sub_frames[i + 1],
                        prev_t_us=sub_prev_t,
                        curr_t_us=sub_curr_t,
                        c_thresh=c_thresh,
                    )

            prev_processed = curr_processed
            prev_t_us = curr_t_us
            frame_idx += 1
    finally:
        cap.release()


def frame_stream(
    source: Union[str, int],
    sensor_size: Tuple[int, int],
    n_bins: int,
    window_ms: int,
    *,
    c_thresh: float = 0.05,
    capture_settings: Optional[dict] = None,
    stride_ms: Optional[int] = None,
    dtype: type = np.float32,
) -> Generator[np.ndarray, None, None]:
    """Yield integrated event frames as ``(n_bins, 2, H, W)`` float32 arrays.

    Each yield covers one time window of ``window_ms`` milliseconds split into
    ``n_bins`` equal bins. Channel 0 = OFF events, channel 1 = ON events.
    Windows are non-overlapping by default; set ``stride_ms < window_ms`` for
    a sliding window.
    """
    H, W = sensor_size
    window_us = window_ms * 1000
    effective_stride_ms = stride_ms if stride_ms is not None else window_ms
    stride_us = effective_stride_ms * 1000
    bin_us = window_us // n_bins

    buffer: list = []
    t_window_start: Optional[int] = None

    for chunk in video_to_event_stream(
        source, c_thresh=c_thresh, sensor_size=sensor_size,
        capture_settings=capture_settings,
    ):
        if chunk.size == 0:
            continue
        if t_window_start is None:
            t_window_start = int(chunk["t"].min())
        buffer.append(chunk)

        # Keep yielding as long as the buffer covers a full window.
        while int(buffer[-1]["t"].max()) - t_window_start >= window_us:
            events = np.concatenate(buffer)
            t_end = t_window_start + window_us
            mask = (events["t"] >= t_window_start) & (events["t"] < t_end)
            w = events[mask]

            frames = np.zeros((n_bins, 2, H, W), dtype=dtype)
            if w.size:
                b = ((w["t"] - t_window_start) // bin_us).clip(0, n_bins - 1).astype(np.int64)
                np.add.at(frames, (b, w["p"].astype(np.int64), w["y"], w["x"]), 1)
            yield frames

            t_window_start += stride_us
            # Drop chunks that are entirely before the new window start.
            buffer = [c for c in buffer if int(c["t"].max()) >= t_window_start]


def write_hdf5(
    path: Union[str, os.PathLike],
    events: np.ndarray,
    sensor_shape: Tuple[int, int],
) -> None:
    """Write events to an HDF5 file using the DVS-Gesture reprocessed layout.

    Layout::

        /events                     (group)
            .attrs["sensor_shape"]  (2,) int  – (height, width)
            /xs   int16   x coords
            /ys   int16   y coords
            /ts   int64   timestamps (µs)
            /ps   int8    polarities ∈ {0, 1}
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        grp = f.create_group("events")
        grp.attrs["sensor_shape"] = np.array(sensor_shape, dtype=np.int64)
        grp.create_dataset("xs", data=events["x"].astype(np.int16), dtype="<i2")
        grp.create_dataset("ys", data=events["y"].astype(np.int16), dtype="<i2")
        grp.create_dataset("ts", data=events["t"].astype(np.int64), dtype="<i8")
        grp.create_dataset("ps", data=events["p"].astype(np.int8), dtype="<i1")
