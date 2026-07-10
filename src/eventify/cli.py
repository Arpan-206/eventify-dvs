"""Command-line entry point for eventify."""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import typer

from eventify._fast import build_log_lut, frame_to_crossing_counts
from eventify.dvs import EVENT_DTYPE, video_to_event_stream, write_hdf5

app = typer.Typer(help="Convert video or webcam feeds into simulated event-camera data.")

# DVS palette (BGR): ON = deep blue, OFF = amber.
_ON_BGR = np.array([180, 70, 0], dtype=np.float32)
_OFF_BGR = np.array([0, 170, 220], dtype=np.float32)


def _events_to_frame(chunk: np.ndarray, h: int, w: int) -> np.ndarray:
    """Render a structured event array into a BGR uint8 frame."""
    on = np.zeros((h, w), dtype=np.float32)
    off = np.zeros((h, w), dtype=np.float32)
    if len(chunk):
        mask_on = chunk["p"] == 1
        np.add.at(on, (chunk["y"][mask_on], chunk["x"][mask_on]), 1)
        np.add.at(off, (chunk["y"][~mask_on], chunk["x"][~mask_on]), 1)

    peak = max(on.max(), off.max(), 1.0)
    intensity_on = np.clip(on / peak, 0.0, 1.0)[..., None]
    intensity_off = np.clip(off / peak, 0.0, 1.0)[..., None]
    img = intensity_on * _ON_BGR + intensity_off * _OFF_BGR
    return np.clip(img, 0, 255).astype(np.uint8)


class _ThreadedCapture:
    """Drops stale frames so the display loop always gets the latest."""

    def __init__(self, source, capture_settings: Optional[dict] = None):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(f"Could not open video source: {source!r}")
        if capture_settings:
            for prop, value in capture_settings.items():
                self._cap.set(prop, value)
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._stop.set()
                break
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
            return None

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
    intensity = np.clip(np.abs(accum) / max_events, 0.0, 1.0)[..., None]
    positive = accum >= 0
    target = np.empty(accum.shape + (3,), dtype=np.float32)
    target[..., 0] = np.where(positive, 180.0, 0.0)
    target[..., 1] = np.where(positive, 70.0, 170.0)
    target[..., 2] = np.where(positive, 0.0, 220.0)
    return np.clip(intensity * target, 0, 255).astype(np.uint8)


@app.command()
def convert(
    input: str = typer.Argument(..., help="Path to the input video file."),
    output: str = typer.Argument(..., help="Path to write the event-rendered video (e.g. out.mp4)."),
    threshold: float = typer.Option(0.05, "--threshold", "-t", help="Log-intensity event threshold."),
):
    """Convert a video file to an event-rendered video."""
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        typer.echo(f"error: could not open input video: {input}", err=True)
        raise typer.Exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        typer.echo(f"error: could not open output for writing: {output}", err=True)
        raise typer.Exit(1)

    count = 0
    try:
        for chunk in video_to_event_stream(input, c_thresh=threshold):
            writer.write(_events_to_frame(chunk, height, width))
            count += 1
    finally:
        writer.release()

    typer.echo(f"wrote {count} event frames to {output}")


@app.command()
def webcam(
    device: int = typer.Option(0, "--device", "-d", help="Webcam device index."),
    threshold: float = typer.Option(0.05, "--threshold", "-t", help="Log-intensity event threshold."),
    width: int = typer.Option(1280, "--width", help="Requested capture width."),
    height: int = typer.Option(720, "--height", help="Requested capture height."),
    fps: float = typer.Option(60.0, "--fps", help="Requested capture FPS."),
    accum_ms: float = typer.Option(80.0, "--accum-ms", help="Event accumulation half-life in ms."),
    max_events: float = typer.Option(8.0, "--max-events", help="Saturation ceiling for accumulated events per pixel."),
):
    """Show a live event stream from the webcam."""
    window = "eventify — press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    log_lut = build_log_lut(eps=1.0)
    half_life_sec = max(accum_ms, 1) / 1000.0

    accum: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    last_tick = time.monotonic()
    frames_shown = 0
    events_seen = 0
    start = time.monotonic()

    capture_settings = {
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FPS: fps,
    }
    cap = _ThreadedCapture(device, capture_settings=capture_settings)
    typer.echo(f"capture opened at {cap.actual_size[0]}x{cap.actual_size[1]}", err=True)

    try:
        while True:
            frame = cap.read(timeout=1.0)
            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                accum = np.zeros(gray.shape, dtype=np.float32)
                continue

            on, off = frame_to_crossing_counts(prev_gray, gray, c_thresh=threshold, log_lut=log_lut)

            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            decay = 0.5 ** (dt / half_life_sec)

            accum *= decay
            accum += on.astype(np.float32)
            accum -= off.astype(np.float32)

            events_seen += int(on.sum()) + int(off.sum())
            cv2.imshow(window, _accum_to_bgr(accum, max_events=max_events))
            frames_shown += 1
            prev_gray = gray

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - start, 1e-6)
    typer.echo(
        f"showed {frames_shown} frames in {elapsed:.1f}s "
        f"({frames_shown / elapsed:.1f} fps display, "
        f"{events_seen / elapsed:,.0f} ev/s)"
    )


@app.command()
def export(
    input: str = typer.Argument(..., help="Path to the input video file."),
    output: str = typer.Argument(..., help="Path to write the events HDF5 file (e.g. out.h5)."),
    threshold: float = typer.Option(0.05, "--threshold", "-t", help="Log-intensity event threshold."),
    sensor_size: Optional[str] = typer.Option(None, "--sensor-size", metavar="W,H", help="Override sensor resolution as 'W,H'."),
    interp: int = typer.Option(0, "--interp", help="Number of interpolated sub-frames between real frames."),
):
    """Export a video's events to a DVS-Gesture-compatible HDF5 file."""
    parsed_sensor_size: Optional[tuple[int, int]] = None
    if sensor_size is not None:
        try:
            w_str, h_str = sensor_size.split(",")
            parsed_sensor_size = (int(w_str), int(h_str))
        except ValueError:
            typer.echo(f"error: --sensor-size must be 'W,H' (got {sensor_size!r})", err=True)
            raise typer.Exit(1)

    chunks = []
    total = 0
    for chunk in video_to_event_stream(
        input,
        c_thresh=threshold,
        sensor_size=parsed_sensor_size,
        interp=interp,
    ):
        chunks.append(chunk)
        total += len(chunk)

    if parsed_sensor_size is not None:
        w, h = parsed_sensor_size
        resolved_shape = (h, w)
    else:
        cap = cv2.VideoCapture(input)
        resolved_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        cap.release()

    events = np.concatenate(chunks) if chunks else np.zeros(0, dtype=EVENT_DTYPE)
    write_hdf5(output, events, sensor_shape=resolved_shape)
    typer.echo(f"wrote {total} events to {output} (sensor_shape={resolved_shape})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
