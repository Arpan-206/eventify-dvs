# eventify

Convert video files or webcam feeds into simulated event-camera (DVS)
data using log-intensity differencing. A clean-room, dependency-light
reimplementation of the core idea behind v2e/ESIM — no CUDA, no PyTorch,
no pretrained models. Just NumPy, OpenCV, and h5py.

Two independent output paths:

1. **Visualization** — signed per-pixel log-delta rendered to color video
   or live webcam preview.
   - **Blue** — brightening pixels (positive delta)
   - **Yellow** — darkening pixels (negative delta)
   - **Gray** — no event
2. **DVS export** — binary-polarity `(x, y, t_µs, p)` event tuples in an
   HDF5 layout compatible with the **DVS128 Gesture** dataset (as
   redistributed by Tonic/SpikingJelly).

Fidelity + performance features:

- **Multi-crossing** — a pixel whose log-delta spans `K` threshold widths
  fires `K` staggered events, matching how a real DVS sensor behaves.
- **Sub-frame interpolation** — with `--interp N`, `N` intermediate
  frames are linearly interpolated between each real frame pair, giving
  finer-grained event timestamps without needing an ML model.
- **Accumulation window** — the live webcam preview fades events over a
  tunable half-life (`--accum-ms`) so slow motion stays visible.
- **LUT-accelerated log** + **threaded capture** — the webcam preview
  uses a 256-entry uint8 lookup table and a drop-newest reader thread,
  measured ~4.7× faster than the reference path at 720p.

## Install

Uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
uv sync
```

After `pip install .` the CLI is available as `eventify`; with `uv sync`
use `uv run eventify ...`.

## CLI

Three subcommands under the `eventify` entry point.

### `eventify convert` — video file → event-visualized video

Reads any video OpenCV can decode, writes an MP4 rendered in the
blue/yellow event style.

```bash
# Defaults (threshold 0.05, per-frame normalized color)
uv run eventify convert input.mp4 events.mp4

# Lower threshold → more events; fixed saturation ceiling
uv run eventify convert input.mp4 events.mp4 --threshold 0.03 --max-delta 1.0
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `input` | — | Path to source video |
| `output` | — | Path to write the rendered MP4 |
| `--threshold` | `0.05` | Log-intensity event threshold |
| `--max-delta` | none | Fixed saturation ceiling. Omit for per-frame normalization |

### `eventify webcam` — live event preview

Opens the default camera, shows the event stream in an OpenCV window.
Press `q` in the window to quit. On macOS, grant camera permission to
your terminal app first (System Settings → Privacy & Security → Camera).

```bash
# Defaults — 1280x720 @ 60 FPS, threaded capture, LUT fast path
uv run eventify webcam

# Snappy, high-sensitivity preview
uv run eventify webcam --threshold 0.03 --accum-ms 40

# 1080p if your camera + CPU handle it
uv run eventify webcam --width 1920 --height 1080

# Longer fade — traces of slow motion linger on-screen
uv run eventify webcam --accum-ms 200 --max-events 12

# Different camera
uv run eventify webcam --device 1
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--device` | `0` | Webcam device index |
| `--threshold` | `0.05` | Log-intensity event threshold |
| `--width` | `1280` | Requested capture width (camera may downgrade) |
| `--height` | `720` | Requested capture height |
| `--fps` | `60` | Requested capture FPS |
| `--accum-ms` | `80` | Event accumulator half-life in ms |
| `--max-events` | `8` | Saturation ceiling for accumulated events per pixel |

On startup, stderr prints `capture opened at WxH` showing the resolution
the camera actually delivered. On quit, stdout prints display FPS and
event rate.

### `eventify export` — video → DVS-Gesture HDF5

Emits binary-polarity DVS events to an HDF5 file. Output layout is
compatible with the DVS128 Gesture dataset as redistributed by Tonic /
SpikingJelly loaders.

```bash
# Native resolution, defaults
uv run eventify export input.mp4 events.h5

# Force DVS128-sized (128x128) sensor grid
uv run eventify export input.mp4 events.h5 --sensor-size 128,128

# Sub-frame interpolation for finer timestamp granularity
uv run eventify export input.mp4 events.h5 --interp 4

# All together — DVS128 grid, low threshold, temporal upsampling
uv run eventify export input.mp4 events.h5 \
    --sensor-size 128,128 --threshold 0.03 --interp 4
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `input` | — | Path to source video |
| `output` | — | Path to write the HDF5 events file |
| `--threshold` | `0.05` | Log-intensity event threshold |
| `--sensor-size` | source video's resolution | Override as `W,H` |
| `--interp` | `0` | Number of interpolated sub-frames between real frames |

## Library

### Visualization path (magnitude preserved)

```python
import cv2
from eventify import frame_to_events, events_to_frame, video_to_events

prev = cv2.imread("a.png", cv2.IMREAD_GRAYSCALE)
curr = cv2.imread("b.png", cv2.IMREAD_GRAYSCALE)
delta = frame_to_events(prev, curr, c_thresh=0.05)  # signed float32 map
img = events_to_frame(delta)                        # BGR uint8 preview

for timestamp, delta in video_to_events("video.mp4"):
    ...
```

### DVS export path (binary polarity)

```python
import numpy as np
from eventify import (
    frame_to_event_tuples,
    video_to_event_stream,
    write_hdf5,
    EVENT_DTYPE,
)

# Per-pair event tuples
events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)
# events["x"], events["y"], events["t"], events["p"]  — p ∈ {0, 1}

# Full stream with sub-frame interp
chunks = list(video_to_event_stream(
    "video.mp4", sensor_size=(128, 128), interp=4
))
all_events = np.concatenate(chunks)
write_hdf5("out.h5", all_events, sensor_shape=(128, 128))
```

### Fast path (LUT-accelerated, for real-time UIs)

```python
from eventify.fast import build_log_lut, frame_to_crossing_counts

lut = build_log_lut(eps=1.0)  # reuse across frames
on_counts, off_counts = frame_to_crossing_counts(
    prev_uint8, curr_uint8, c_thresh=0.05, log_lut=lut
)
# on_counts, off_counts are 2D int16 arrays — how many crossings per pixel.
```

## API reference

### Visualization

- **`frame_to_events(prev, curr, c_thresh=0.05, eps=1.0)`** — returns a
  2D `float32` array of `log(curr + eps) − log(prev + eps)` with values
  below `c_thresh` in magnitude zeroed. Positive = brightening,
  negative = darkening.
- **`events_to_frame(delta, max_delta=None)`** — returns an `H×W×3` BGR
  `uint8` image. Per-frame normalized by default; pass `max_delta` for a
  fixed saturation ceiling (magnitudes are clipped).
- **`video_to_events(source, c_thresh=0.05, grayscale=True)`** —
  generator yielding `(timestamp_seconds, delta)` tuples.

### DVS export

- **`frame_to_event_tuples(prev, curr, prev_t_us, curr_t_us, c_thresh=0.05, eps=1.0, sensor_size=None)`** —
  returns a NumPy structured array of dtype `EVENT_DTYPE` with fields
  `(x: i2, y: i2, t: i8, p: i1)`. Polarity is binary (0 = OFF,
  1 = ON). **Multi-crossing**: a pixel whose log-delta spans `K`
  thresholds emits `K` events, uniformly staggered in the interval so
  their timestamps are strictly monotonic.
- **`interpolate_frames(prev, curr, n_intermediate)`** — linearly
  interpolates `n_intermediate` frames between two endpoints, returning
  a list of `n_intermediate + 2` frames.
- **`video_to_event_stream(source, c_thresh=0.05, sensor_size=None, interp=0, capture_settings=None)`** —
  generator yielding one structured event array per (sub-)frame-pair.
  Timestamps across chunks are monotonic microseconds. Pass
  `sensor_size=(w, h)` to resize frames, `interp=N` to insert `N`
  interpolated frames between each real pair, or `capture_settings` to
  pass OpenCV `CAP_PROP_*` overrides to a live device.
- **`write_hdf5(path, events, sensor_shape)`** — writes events in the
  DVS-Gesture reprocessed layout:

  ```
  /events                          (group)
      .attrs["sensor_shape"]  (2,) i8   — (height, width)
      /xs   i2   x coords
      /ys   i2   y coords
      /ts   i8   timestamps (µs)
      /ps   i1   polarities ∈ {0, 1}
  ```

### Fast path

- **`build_log_lut(eps=1.0)`** — precomputes `log(x + eps)` for
  `x ∈ [0, 255]` as a 256-entry float32 array.
- **`frame_to_crossing_counts(prev_uint8, curr_uint8, c_thresh=0.05, log_lut=None)`** —
  returns `(on_counts, off_counts)`, two 2D int16 arrays of per-pixel
  crossing counts. Requires uint8 grayscale input. If `log_lut` is
  omitted, it's built on every call — for hot loops, pass a shared LUT.

## Tests

```bash
uv run pytest
```

72 tests covering the reference path, the DVS export path, the fast
path, HDF5 layout, and CLI-adjacent primitives.

## License

MIT — see [LICENSE](LICENSE).
