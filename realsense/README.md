# RealSense Real-Time Anonymization

Real-time person anonymization running directly from an **Intel RealSense D415 / D435** camera using the **LA3D** area-based approach — no depth model inference required, so both scripts run comfortably on **CPU only**.

---

## Scripts

| Script | Target | Formula | Output file |
|---|---|---|---|
| `anon_body.py` | Full body silhouette | A1 — `r = α·log(body_box_area / frame_area × 100)` | `anonymized_body.mp4` |
| `anon_head.py` | Head region only | A3 — `r = α·log(head_box_area / frame_area × 5000)` | `anonymized_head.mp4` |

Both scripts use:
- **YOLOv8n-seg** for body segmentation masks (silhouette-shaped anonymization, no rectangular boxes)
- **YOLOv8n-pose** for head keypoints (COCO **0–4**: nose, eyes, ears — shoulders excluded so the head box stays tight)
- The head script uses the **head box ∩ body mask** so background inside the box is not blurred

**Standalone layout:** `anon_body.py` / `anon_head.py` only need this `realsense/` directory. Shared logic lives in `realsense/anon_support/` (a small vendored copy of the main repo’s detection / filters / formulas). You do **not** need the parent `depth-aware-anon` repo on `PYTHONPATH`.

---

## Prerequisites

### 1 · Intel RealSense SDK
Install `librealsense2` from Intel's releases before installing the Python wrapper:

```
https://github.com/IntelRealSense/librealsense/releases
```

Then install the Python wrapper:

```bash
pip install pyrealsense2
```

### 2 · Python packages

```bash
cd /path/to/realsense   # this folder only
pip install -r requirements.txt
```

---

## Usage

### Body anonymization (A1)

```bash
cd realsense
python anon_body.py
```

### Head anonymization (A3)

```bash
cd realsense
python anon_head.py
```

You can copy the whole `realsense/` tree elsewhere (e.g. another machine with only RealSense + Python); keep `anon_support/` next to the scripts.

---

## Window controls

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Start / stop recording to `.mp4` |
| `+` / `-` | Increase / decrease blur strength (`alpha`) |
| `d` | Toggle depth-map panel on the right |
| `b` | *(head script only)* Toggle body & head bounding-box overlay |

---

## Tuning

Edit the constants at the top of each script:

| Constant | Default | Effect |
|---|---|---|
| `ALPHA_BODY` / `ALPHA_HEAD` | 1.0 / 1.2 | Log-radius scale — higher = stronger blur |
| `BLUR_KERNEL_BASE` | 13 | Kernel size multiplier (kernel = ⌈r × base⌉, rounded to odd) |
| `DILATION_PX` | 10 / 8 | Extra padding around the person mask in pixels |
| `YOLO_SEG_MODEL` | `yolov8n-seg.pt` | Replace with `yolov8s-seg.pt` for more accuracy at lower FPS |
| `CONF_THRESHOLD` | 0.30 | YOLO detection confidence threshold |
| `STREAM_WIDTH/HEIGHT` | 1280 × 720 | Camera resolution (reduce to 640×480 for faster CPU throughput) |
| `DISPLAY_SCALE` | 0.75 | Fraction of full resolution shown in the display window |

---

## Performance tips (CPU)

- Switch to **640 × 480** resolution (`STREAM_WIDTH = 640`, `STREAM_HEIGHT = 480`) for ~2× speed-up with minimal quality loss.
- Use `yolov8n` (nano) models — already the default. Avoid `yolov8m` or larger on CPU.
- Close other GPU-heavy applications; YOLOv8 on CPU uses all available cores via OpenMP.
- Expected throughput on a modern laptop CPU: **4–10 FPS** at 1280×720, **10–20 FPS** at 640×480.

---

## Other files in this folder

| Path | Purpose |
|---|---|
| `anon_support/` | Vendored detection / filters / formulas for **standalone** runs (no parent repo). |
| `stream.py` | Colour + depth side-by-side viewer (hardware check). |
| `stream_raw.py` | Minimal depth read at centre pixel. |
| `save_video.py` | Record colour + raw depth `.npy` to `realsense_validation_dataset/`. |
| `save_file_2.py` | RAM-buffered recording with HSV depth preview. |
| `requirements.txt` | Pip deps for this directory. |

If you only care about live anonymization, **`stream*.py`** and **`save*.py`** are optional utilities — you can delete them to shrink the folder; **`anon_support/`** must stay with **`anon_*.py`**.
