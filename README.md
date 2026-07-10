# eventify-dvs

Convert video files or webcam feeds into simulated event-camera (DVS) data using log-intensity differencing. A clean-room, dependency-light reimplementation of the core idea behind v2e/ESIM — no CUDA, no PyTorch, no pretrained models. Just NumPy, OpenCV, and h5py.

## Install

```bash
pip install eventify-dvs
```

The CLI entry point is `eventify`. With `uv`:

```bash
uv add eventify-dvs
uv run eventify --help
```

## CLI

Three subcommands under the `eventify` entry point.

### `eventify webcam` — live event preview

Opens the default camera and shows the event stream in an OpenCV window. Press `q` to quit. On macOS, grant camera permission to your terminal app first (System Settings → Privacy & Security → Camera).

```bash
# Defaults — 1280x720 @ 60 FPS
eventify webcam

# Snappy, high-sensitivity preview
eventify webcam --threshold 0.03 --accum-ms 40

# Different camera
eventify webcam --device 1
```

| Flag | Default | Purpose |
|---|---|---|
| `--device` | `0` | Webcam device index |
| `--threshold` | `0.05` | Log-intensity event threshold |
| `--width` | `1280` | Requested capture width |
| `--height` | `720` | Requested capture height |
| `--fps` | `60` | Requested capture FPS |
| `--accum-ms` | `80` | Event accumulator half-life in ms |
| `--max-events` | `8` | Saturation ceiling for accumulated events per pixel |

### `eventify convert` — video file → event-visualized video

```bash
eventify convert input.mp4 events.mp4
eventify convert input.mp4 events.mp4 --threshold 0.03
```

| Flag | Default | Purpose |
|---|---|---|
| `input` | — | Path to source video |
| `output` | — | Path to write the rendered MP4 |
| `--threshold` | `0.05` | Log-intensity event threshold |

### `eventify export` — video → DVS-Gesture HDF5

Emits binary-polarity DVS events to an HDF5 file compatible with the DVS128 Gesture dataset layout (Tonic / SpikingJelly loaders).

```bash
eventify export input.mp4 events.h5
eventify export input.mp4 events.h5 --sensor-size 128,128 --interp 4
```

| Flag | Default | Purpose |
|---|---|---|
| `input` | — | Path to source video |
| `output` | — | Path to write the HDF5 events file |
| `--threshold` | `0.05` | Log-intensity event threshold |
| `--sensor-size` | source resolution | Override as `W,H` |
| `--interp` | `0` | Interpolated sub-frames between real frames |

## Library

```python
import numpy as np
from eventify import (
    frame_to_event_tuples,
    video_to_event_stream,
    interpolate_frames,
    write_hdf5,
    EVENT_DTYPE,
)

# Per-frame-pair event tuples
events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)
# events["x"], events["y"], events["t"], events["p"]  — p ∈ {0, 1}

# Full stream from a video file
chunks = list(video_to_event_stream("video.mp4", sensor_size=(128, 128), interp=4))
all_events = np.concatenate(chunks)
write_hdf5("out.h5", all_events, sensor_shape=(128, 128))
```

## API reference

- **`frame_to_event_tuples(prev, curr, prev_t_us, curr_t_us, c_thresh=0.05, eps=1.0, sensor_size=None)`** —
  returns a NumPy structured array with dtype `EVENT_DTYPE` and fields
  `(x: i2, y: i2, t: i8, p: i1)`. Polarity is binary (0 = OFF, 1 = ON).
  A pixel whose log-delta spans `K` thresholds emits `K` events, uniformly
  staggered across the interval.

- **`video_to_event_stream(source, c_thresh=0.05, sensor_size=None, interp=0, capture_settings=None)`** —
  generator yielding one structured event array per (sub-)frame-pair.
  Timestamps are monotonic microseconds.

- **`interpolate_frames(prev, curr, n_intermediate)`** —
  linearly interpolates `n_intermediate` frames between two endpoints,
  returning a list of `n_intermediate + 2` frames.

- **`write_hdf5(path, events, sensor_shape)`** — writes events in the
  DVS-Gesture reprocessed layout:

  ```
  /events
      .attrs["sensor_shape"]  (height, width)
      /xs   i2
      /ys   i2
      /ts   i8   microseconds
      /ps   i1   ∈ {0, 1}
  ```

- **`EVENT_DTYPE`** — NumPy structured dtype `[("x", "<i2"), ("y", "<i2"), ("t", "<i8"), ("p", "<i1")]`.

## Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
