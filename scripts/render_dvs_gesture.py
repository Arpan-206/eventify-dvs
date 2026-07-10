"""Render a real DVS128 Gesture .aedat recording through eventify's palette.

Parses raw (x, y, t_µs, polarity) events from an AEDAT 3.1 file (the
format the DVS128 Gesture dataset ships in), bins them into short time
windows, and renders each window through an exponentially-fading
accumulator so that motion trails persist on screen — the look of the
reference DVS visualization.

Uses SpikingJelly's ``load_aedat_v3`` framing logic (MIT-licensed,
credited inline) but with a vectorized event-decode loop for a ~100×
speedup on 60 MB files.

Usage:
    uv run python scripts/render_dvs_gesture.py \\
        dvs_gesture_data/DvsGesture/user01_fluorescent.aedat \\
        user01_events.mp4 \\
        [--start-us 80000000 --duration-s 10 --bin-ms 20 --fade-ms 100 --upscale 4]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

# --- AEDAT 3.1 parser -------------------------------------------------------
#
# Header framing adapted from SpikingJelly's load_aedat_v3
# (https://github.com/fangwei123456/spikingjelly, MIT). The inner event
# decode loop was rewritten with NumPy so we don't iterate in Python over
# millions of events.


def load_aedat_v3(path: Path) -> dict:
    """Parse an AEDAT 3.1 file into a dict of NumPy arrays.

    Returns ``{"t": µs int64, "x": int16, "y": int16, "p": uint8}``.
    """
    ts_list: list = []
    xs_list: list = []
    ys_list: list = []
    ps_list: list = []

    with open(path, "rb") as f:
        # Skip ASCII header — lines start with '#', ends with #!END-HEADER.
        line = f.readline()
        while line.startswith(b"#"):
            if line == b"#!END-HEADER\r\n":
                break
            line = f.readline()

        while True:
            header = f.read(28)
            if not header or len(header) < 28:
                break

            e_type = struct.unpack("H", header[0:2])[0]
            e_size = struct.unpack("I", header[4:8])[0]
            e_tsoverflow = struct.unpack("I", header[12:16])[0]
            e_capacity = struct.unpack("I", header[16:20])[0]

            data_length = e_capacity * e_size
            data = f.read(data_length)

            if e_type != 1:
                # Only polarity events (type 1) are relevant here.
                continue

            # AEDAT 3.1 polarity event: 8 bytes = 4-byte aer_data + 4-byte timestamp.
            # Vectorized decode: interpret the whole packet as a uint32 pair per event.
            packet = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
            aer = packet[:, 0]
            ts_low = packet[:, 1].astype(np.int64)

            xs = ((aer >> 17) & 0x00007FFF).astype(np.int16)
            ys = ((aer >> 2) & 0x00007FFF).astype(np.int16)
            ps = ((aer >> 1) & 0x00000001).astype(np.uint8)
            ts = ts_low | (np.int64(e_tsoverflow) << 31)

            ts_list.append(ts)
            xs_list.append(xs)
            ys_list.append(ys)
            ps_list.append(ps)

    if not ts_list:
        return {"t": np.zeros(0, np.int64), "x": np.zeros(0, np.int16),
                "y": np.zeros(0, np.int16), "p": np.zeros(0, np.uint8)}

    return {
        "t": np.concatenate(ts_list),
        "x": np.concatenate(xs_list),
        "y": np.concatenate(ys_list),
        "p": np.concatenate(ps_list),
    }


# --- Renderer ---------------------------------------------------------------


# DVS palette (BGR). Matches eventify.core.
_ON = np.array([180, 70, 0], dtype=np.float32)
_OFF = np.array([0, 170, 220], dtype=np.float32)


def render_accum(accum: np.ndarray, max_events: float) -> np.ndarray:
    """Signed accumulator → BGR uint8 image (black bg, blue ON, amber OFF)."""
    intensity = np.clip(np.abs(accum) / max_events, 0.0, 1.0)[..., None]
    positive = accum >= 0
    target = np.empty(accum.shape + (3,), dtype=np.float32)
    target[..., 0] = np.where(positive, _ON[0], _OFF[0])
    target[..., 1] = np.where(positive, _ON[1], _OFF[1])
    target[..., 2] = np.where(positive, _ON[2], _OFF[2])
    img = intensity * target
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("aedat", help="Path to .aedat file")
    p.add_argument("out", help="Output MP4 path")
    p.add_argument("--start-us", type=int, default=None, help="Start timestamp in µs (default: first event)")
    p.add_argument("--duration-s", type=float, default=10.0, help="Duration to render (default: 10 s)")
    p.add_argument("--bin-ms", type=float, default=20.0, help="Time window per output frame in ms (default: 20 → 50 fps)")
    p.add_argument("--fade-ms", type=float, default=100.0, help="Accumulator half-life in ms (default: 100)")
    p.add_argument("--upscale", type=int, default=4, help="Upscale factor for the 128×128 sensor grid (default: 4)")
    p.add_argument("--max-events", type=float, default=6.0, help="Saturation ceiling per pixel (default: 6)")
    args = p.parse_args()

    aedat_path = Path(args.aedat)
    print(f"parsing {aedat_path} ({aedat_path.stat().st_size / 1e6:.1f} MB) ...")
    ev = load_aedat_v3(aedat_path)
    n = len(ev["t"])
    if n == 0:
        raise SystemExit("no events found in file")

    t0_file = int(ev["t"].min())
    t1_file = int(ev["t"].max())
    span_s = (t1_file - t0_file) / 1e6
    print(f"  {n:,} events over {span_s:.1f} s (t0={t0_file}, t1={t1_file})")

    start_us = args.start_us if args.start_us is not None else t0_file
    end_us = start_us + int(args.duration_s * 1e6)

    mask = (ev["t"] >= start_us) & (ev["t"] < end_us)
    t = ev["t"][mask]
    x = ev["x"][mask]
    y = ev["y"][mask]
    pol = ev["p"][mask]
    if len(t) == 0:
        raise SystemExit(f"no events in [{start_us}, {end_us})")
    print(f"  {len(t):,} events in selected {args.duration_s:.1f} s window")

    # Sensor is 128×128 for DVS128, but be defensive against coords outside that.
    H = int(max(128, y.max() + 1))
    W = int(max(128, x.max() + 1))
    print(f"  sensor grid: {H}×{W}")

    bin_us = int(args.bin_ms * 1000)
    n_bins = (end_us - start_us + bin_us - 1) // bin_us
    fps = 1000.0 / args.bin_ms
    up = args.upscale
    h_out, w_out = H * up, W * up

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out), True)
    if not writer.isOpened():
        raise SystemExit(f"could not open writer: {args.out}")

    accum = np.zeros((H, W), dtype=np.float32)
    # Precompute per-bin event slice boundaries (events are already time-sorted).
    bin_ids = ((t - start_us) // bin_us).astype(np.int64)
    slice_starts = np.searchsorted(bin_ids, np.arange(n_bins))
    slice_ends = np.searchsorted(bin_ids, np.arange(1, n_bins + 1))

    half_life_bins = args.fade_ms / args.bin_ms
    decay = 0.5 ** (1.0 / max(half_life_bins, 1e-6))

    try:
        for b in range(n_bins):
            accum *= decay
            s, e = slice_starts[b], slice_ends[b]
            if e > s:
                bx = x[s:e]
                by = y[s:e]
                bp = pol[s:e]
                # Signed splat via bincount, faster than np.add.at.
                signed = np.where(bp == 1, 1.0, -1.0).astype(np.float32)
                flat_idx = by.astype(np.int64) * W + bx.astype(np.int64)
                delta = np.bincount(flat_idx, weights=signed, minlength=H * W)
                accum += delta.reshape(H, W).astype(np.float32)

            frame = render_accum(accum, max_events=args.max_events)
            big = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
            # Time overlay
            t_sec = (start_us + b * bin_us - t0_file) / 1e6
            cv2.putText(big, f"t={t_sec:5.2f}s  bin {b + 1}/{n_bins}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(big)
    finally:
        writer.release()

    print(f"wrote {n_bins} frames to {args.out} at {fps:.1f} fps "
          f"(bin={args.bin_ms}ms, fade={args.fade_ms}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
