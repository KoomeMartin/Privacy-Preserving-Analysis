"""anon_body.py  —  Real-time body anonymization (LA3D A1 approach)
====================================================================
Streams from an Intel RealSense D415/D435 camera, detects people with
YOLOv8-seg, and applies a **Gaussian blur** to each person's full body
silhouette.  The blur radius is driven by the LA3D A1 formula:

    r = alpha_body * log( body_box_area / frame_area * 100 )

This is the "area-only, no depth" baseline — equivalent to LA3D's
original body-anonymization approach.

Controls (OpenCV window)
------------------------
  q       quit
  r       toggle recording to anonymized_body.mp4
  +/-     increase / decrease blur strength (alpha_body)
  d       toggle depth-map overlay (right panel)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

_RS_ROOT = Path(__file__).resolve().parent
if str(_RS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RS_ROOT))

from anon_support.config import AnonymizationConfig
from anon_support.detection import detect_people
from anon_support.filters import apply_masked_filter
from anon_support.formulas import a1_body

# ─────────────────────────────────────────────────────────────────────────────
# Tuning — edit these to change behaviour without touching the logic
# ─────────────────────────────────────────────────────────────────────────────
STREAM_WIDTH   = 1280
STREAM_HEIGHT  = 720
STREAM_FPS     = 30
DEVICE         = "cpu"           # CPU only — no CUDA needed
YOLO_SEG_MODEL = "yolov8n-seg.pt"  # nano for speed on CPU; try yolov8s-seg.pt for more accuracy
YOLO_POSE_MODEL = "yolov8n-pose.pt"
ALPHA_BODY     = 1.0             # log-radius scale; increase for stronger blur
BLUR_KERNEL_BASE = 13            # kernel = ceil(r * base), must resolve to odd
DILATION_PX    = 10              # mask edge padding in pixels
CONF_THRESHOLD = 0.30
DISPLAY_SCALE  = 0.75            # scale for the display window (keep it manageable)
# ─────────────────────────────────────────────────────────────────────────────

def _a1_radius(body_bbox, mask, frame_h: int, frame_w: int, alpha: float) -> float:
    """LA3D A1: log-radius from body segmentation mask area fraction.
    
    r = alpha * log(100 * body_mask_area / frame_area)
    """
    mask_area = float(np.sum(mask > 0))
    area_frac = mask_area / max(frame_h * frame_w, 1)
    return a1_body(area_frac, alpha)


def _overlay_info(frame: np.ndarray, fps: float, alpha: float, recording: bool, r_values: list) -> None:
    h, w = frame.shape[:2]
    bar = np.zeros((60, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    rec_tag = "  ● REC" if recording else ""
    text = (f"FPS {fps:4.1f}  |  alpha={alpha:.2f}  |  people={len(r_values)}"
            f"  |  [+/-] strength  [r] record  [q] quit{rec_tag}")
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220) if not recording else (80, 80, 255), 1, cv2.LINE_AA)
    
    # Display r values for each person
    if r_values:
        r_text = "  ".join([f"r{i+1}={r:.2f}" for i, r in enumerate(r_values)])
        cv2.putText(bar, r_text, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 200, 255), 1, cv2.LINE_AA)
    
    frame[:60] = bar


def main() -> None:
    # ── Load YOLO models (CPU) ────────────────────────────────────────────────
    print("Loading YOLO models on CPU …")
    from ultralytics import YOLO
    seg_model  = YOLO(YOLO_SEG_MODEL)
    pose_model = YOLO(YOLO_POSE_MODEL)
    seg_model.to("cpu")
    pose_model.to("cpu")

    anon_cfg = AnonymizationConfig(
        blur_kernel_base=BLUR_KERNEL_BASE,
        dilation_px=DILATION_PX,
    )

    # ── RealSense pipeline ────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)
    profile  = pipeline.start(config)
    align    = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"  Camera ready  |  depth scale = {depth_scale}")

    alpha      = ALPHA_BODY
    recording  = False
    writer     = None
    display_w  = int(STREAM_WIDTH  * DISPLAY_SCALE)
    display_h  = int(STREAM_HEIGHT * DISPLAY_SCALE)
    show_depth = False

    # FPS tracking
    t_prev  = time.perf_counter()
    fps_val = 0.0

    print("Window open — press 'q' to quit, 'r' to record, '+'/'-' to tune blur.")

    try:
        while True:
            frames         = pipeline.wait_for_frames(10_000)
            aligned        = align.process(frames)
            depth_frame    = aligned.get_depth_frame()
            color_frame    = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr  = np.asanyarray(color_frame.get_data()).copy()
            depth_raw  = np.asanyarray(depth_frame.get_data()).copy()   # uint16 mm
            h, w       = color_bgr.shape[:2]

            # ── Detect people ─────────────────────────────────────────────────
            people = detect_people(color_bgr, seg_model, pose_model, conf_threshold=CONF_THRESHOLD)

            # ── Apply A1 blur to each person silhouette ───────────────────────
            anon_frame = color_bgr.copy()
            r_values = []
            for person in people:
                r = _a1_radius(person.bbox_xyxy, person.mask, h, w, alpha)
                r_values.append(r)
                anon_frame = apply_masked_filter(
                    anon_frame,
                    person.mask.astype(bool),
                    method="blur",
                    radius=r,
                    depth_value=1.0,   # A1 ignores depth
                    cfg=anon_cfg,
                )

            # ── FPS ───────────────────────────────────────────────────────────
            t_now   = time.perf_counter()
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev  = t_now

            # ── Build display frame ───────────────────────────────────────────
            _overlay_info(anon_frame, fps_val, alpha, recording, r_values)

            if show_depth:
                depth_coloured = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_INFERNO
                )
                display_frame = np.hstack([anon_frame, depth_coloured])
                dw = int(display_w * 2)
            else:
                display_frame = anon_frame
                dw = display_w

            cv2.imshow("LA3D Body Anonymization  [RealSense]",
                       cv2.resize(display_frame, (dw, display_h)))

            # ── Record ────────────────────────────────────────────────────────
            if recording:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter("anonymized_body.mp4", fourcc, STREAM_FPS, (w, h))
                writer.write(anon_frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("Recording saved to anonymized_body.mp4")
                else:
                    print("Recording started …")
            elif key == ord("+") or key == ord("="):
                alpha = min(alpha + 0.1, 5.0)
            elif key == ord("-"):
                alpha = max(alpha - 0.1, 0.1)
            elif key == ord("d"):
                show_depth = not show_depth

    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
