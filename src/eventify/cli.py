"""Command-line entry point for eventify."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from eventify.core import events_to_frame, video_to_events
from eventify.dvs import EVENT_DTYPE, video_to_event_stream, write_hdf5
from eventify.fast import build_log_lut, frame_to_crossing_counts


def _parse_sensor_size(spec: str) -> tuple[int, int]:
    try:
        w_str, h_str = spec.split(",")
        return int(w_str), int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--sensor-size must be 'W,H' (got {spec!r})"
        ) from exc


def _capture_settings_from_args(args: argparse.Namespace) -> Optional[dict]:
    """Collect CAP_PROP_* overrides from --width/--height/--fps."""
    props = {}
    if getattr(args, "width", None):
        props[cv2.CAP_PROP_FRAME_WIDTH] = args.width
    if getattr(args, "height", None):
        props[cv2.CAP_PROP_FRAME_HEIGHT] = args.height
    if getattr(args, "fps", None):
        props[cv2.CAP_PROP_FPS] = args.fps
    return props or None


def _convert(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"error: could not open input video: {args.input}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        print(f"error: could not open output for writing: {args.output}", file=sys.stderr)
        return 1

    count = 0
    try:
        for _, delta in video_to_events(args.input, c_thresh=args.threshold):
            frame = events_to_frame(delta, max_delta=args.max_delta)
            writer.write(frame)
            count += 1
    finally:
        writer.release()

    print(f"wrote {count} event frames to {args.output}")
    return 0


class _ThreadedCapture:
    """Camera reader that always yields the latest frame, dropping stale ones.

    OpenCV's cap.read() buffers internally and blocks — reading in the main
    loop introduces a queue-shaped latency between real motion and display.
    Reading in a background thread with a single-slot drop-newest queue means
    the consumer always sees the freshest frame available at read time.
    """

    def __init__(self, source, capture_settings: Optional[dict] = None):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(f"Could not open video source: {source!r}")
        if capture_settings:
            for prop, value in capture_settings.items():
                self._cap.set(prop, value)
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._stop.set()
                break
            # Drop the previously queued frame if the consumer hasn't taken it.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None if self._stop.is_set() else None

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()

    @property
    def actual_size(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


def _accum_to_bgr(accum: np.ndarray, max_events: float) -> np.ndarray:
    """Render the accumulator to BGR: blue for +, yellow for -, gray otherwise."""
    intensity = np.clip(np.abs(accum) / max_events, 0.0, 1.0)[..., None]
    # Vectorized: gray + intensity * (target - gray), where target ∈ {blue, yellow}.
    positive = accum >= 0
    # Precompute BGR triplets so we don't rebuild them every frame.
    target = np.empty(accum.shape + (3,), dtype=np.float32)
    target[..., 0] = np.where(positive, 255.0, 0.0)    # B
    target[..., 1] = np.where(positive, 0.0, 255.0)    # G
    target[..., 2] = np.where(positive, 0.0, 255.0)    # R
    gray = 128.0
    img = gray + intensity * (target - gray)
    return np.clip(img, 0, 255).astype(np.uint8)


def _webcam(args: argparse.Namespace) -> int:
    window = "eventify — press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    log_lut = build_log_lut(eps=1.0)
    half_life_sec = max(args.accum_ms, 1) / 1000.0

    accum: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    last_tick = time.monotonic()

    frames_shown = 0
    events_seen = 0
    start = time.monotonic()

    cap = _ThreadedCapture(args.device, capture_settings=_capture_settings_from_args(args))
    print(f"capture opened at {cap.actual_size[0]}x{cap.actual_size[1]}", file=sys.stderr)

    try:
        while True:
            frame = cap.read(timeout=1.0)
            if frame is None:
                # End of stream (webcam disconnected or file ended).
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # uint8

            if prev_gray is None:
                prev_gray = gray
                accum = np.zeros(gray.shape, dtype=np.float32)
                continue

            on, off = frame_to_crossing_counts(
                prev_gray, gray, c_thresh=args.threshold, log_lut=log_lut
            )

            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            decay = 0.5 ** (dt / half_life_sec)

            # In-place fade + splat: both O(n_pixels) vectorized.
            accum *= decay
            accum += on.astype(np.float32)
            accum -= off.astype(np.float32)

            events_seen += int(on.sum()) + int(off.sum())

            img = _accum_to_bgr(accum, max_events=args.max_events)
            cv2.imshow(window, img)
            frames_shown += 1
            prev_gray = gray

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - start, 1e-6)
    print(
        f"showed {frames_shown} frames in {elapsed:.1f}s "
        f"({frames_shown / elapsed:.1f} fps display, "
        f"{events_seen / elapsed:,.0f} ev/s)"
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    chunks = []
    total = 0
    for chunk in video_to_event_stream(
        args.input,
        c_thresh=args.threshold,
        sensor_size=args.sensor_size,
        interp=args.interp,
    ):
        chunks.append(chunk)
        total += len(chunk)

    if args.sensor_size is not None:
        w, h = args.sensor_size
        resolved_shape = (h, w)
    else:
        cap = cv2.VideoCapture(args.input)
        resolved_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        cap.release()

    events = np.concatenate(chunks) if chunks else np.zeros(0, dtype=EVENT_DTYPE)
    write_hdf5(args.output, events, sensor_shape=resolved_shape)
    print(f"wrote {total} events to {args.output} (sensor_shape={resolved_shape})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eventify",
        description="Convert video or webcam feeds into simulated event-camera data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert a video file to an event-rendered video.")
    convert.add_argument("input", help="Path to the input video file.")
    convert.add_argument("output", help="Path to write the event-rendered video (e.g. out.mp4).")
    convert.add_argument("--threshold", type=float, default=0.05, help="Log-intensity event threshold (default: 0.05).")
    convert.add_argument("--max-delta", type=float, default=None, help="Fixed saturation ceiling; omit for per-frame normalization.")
    convert.set_defaults(func=_convert)

    webcam = sub.add_parser("webcam", help="Show live event stream from the webcam.")
    webcam.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0).")
    webcam.add_argument("--threshold", type=float, default=0.05, help="Log-intensity event threshold (default: 0.05).")
    webcam.add_argument("--width", type=int, default=1280, help="Requested capture width (default: 1280).")
    webcam.add_argument("--height", type=int, default=720, help="Requested capture height (default: 720).")
    webcam.add_argument("--fps", type=float, default=60.0, help="Requested capture FPS (default: 60).")
    webcam.add_argument("--accum-ms", type=float, default=80.0, help="Event accumulation half-life in ms (default: 80).")
    webcam.add_argument("--max-events", type=float, default=8.0, help="Saturation ceiling for accumulated events per pixel (default: 8).")
    webcam.set_defaults(func=_webcam)

    export = sub.add_parser(
        "export",
        help="Export a video's events to a DVS-Gesture-compatible HDF5 file.",
    )
    export.add_argument("input", help="Path to the input video file.")
    export.add_argument("output", help="Path to write the events HDF5 file (e.g. out.h5).")
    export.add_argument("--threshold", type=float, default=0.05, help="Log-intensity event threshold (default: 0.05).")
    export.add_argument(
        "--sensor-size",
        type=_parse_sensor_size,
        default=None,
        metavar="W,H",
        help="Override sensor resolution as 'W,H' (default: source video's native resolution).",
    )
    export.add_argument("--interp", type=int, default=0, help="Number of interpolated sub-frames between real frames (default: 0).")
    export.set_defaults(func=_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
