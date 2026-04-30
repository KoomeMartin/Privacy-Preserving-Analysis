"""anon_head.py  —  Real-time head anonymization (LA3D A3 approach)
====================================================================
Streams from an Intel RealSense D415/D435 camera, detects people with
YOLOv8-pose (for head keypoints) + YOLOv8-seg (for body segmentation),
and applies a **Gaussian blur** restricted to the head silhouette region.

The blur radius follows the LA3D A3 formula:

    r = alpha_head * log( head_box_area / frame_area * 5000 )

where the head box is derived from COCO keypoints **0–4** (nose, eyes, ears)
intersected with the body segmentation mask so the anonymization follows
the head silhouette rather than a loose rectangle.

Controls (OpenCV window)
------------------------
  q       quit
  r       toggle recording to anonymized_head.mp4
  +/-     increase / decrease blur strength (alpha_head)
  b       toggle body bbox overlay
  d       toggle depth-map overlay (right panel)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# This folder is self-contained: add ``realsense/`` (this file's directory) on sys.path.
_RS_ROOT = Path(__file__).resolve().parent
if str(_RS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RS_ROOT))

from anon_support.config import AnonymizationConfig
from anon_support.detection import box_area, detect_people
from anon_support.filters import apply_masked_filter
from anon_support.formulas import a1_body, a3_head

# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────
STREAM_WIDTH    = 1280
STREAM_HEIGHT   = 720
STREAM_FPS      = 30
YOLO_SEG_MODEL  = "yolov8n-seg.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"
ALPHA_HEAD      = 0.8            # log-radius scale for head; slightly higher than body
BLUR_KERNEL_BASE = 13
DILATION_PX     = 8              # slightly tighter dilation than body
CONF_THRESHOLD  = 0.30
DISPLAY_SCALE   = 0.75
# ─────────────────────────────────────────────────────────────────────────────


def _a3_radius(head_bbox, frame_h: int, frame_w: int, alpha: float) -> float:
    """LA3D A3: log-radius from head bounding-box area fraction.
    
    r = alpha * log(5000 * head_bbox_area / frame_area)
    
    Head area is typically 1-2% of frame, so multiplier 5000 ensures
    log term is positive and comparable to A1 (body area ~20% → multiplier 100).
    """
    head_area = box_area(head_bbox)
    area_frac = head_area / max(frame_h * frame_w, 1)
    return a3_head(area_frac, alpha)





def _draw_boxes(frame: np.ndarray, people, show_body: bool) -> np.ndarray:
    canvas = frame.copy()
    for person in people:
        if show_body:
            x1, y1, x2, y2 = person.bbox_xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 220, 50), 1)
        if person.head_box_xyxy is not None:
            hx1, hy1, hx2, hy2 = person.head_box_xyxy
            cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), (30, 160, 255), 2)
    return canvas


def _overlay_info(frame: np.ndarray, fps: float, alpha: float, recording: bool, r_values: list) -> None:
    w = frame.shape[1]
    bar = np.zeros((60, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    rec_tag = "  ● REC" if recording else ""
    text = (f"FPS {fps:4.1f}  |  alpha_head={alpha:.2f}  |  people={len(r_values)}"
            f"  |  [+/-] strength  [b] bbox  [r] record  [q] quit{rec_tag}")
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220) if not recording else (80, 80, 255), 1, cv2.LINE_AA)
    
    # Display r values for each person
    if r_values:
        r_text = "  ".join([f"r{i+1}={r:.2f}" for i, r in enumerate(r_values)])
        cv2.putText(bar, r_text, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 200, 255), 1, cv2.LINE_AA)
    
    frame[:60] = bar


def main() -> None:
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

    # ── RealSense ─────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)
    profile  = pipeline.start(config)
    align    = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"  Camera ready  |  depth scale = {depth_scale}")

    alpha      = ALPHA_HEAD
    recording  = False
    writer     = None
    show_body  = False
    show_depth = False
    display_w  = int(STREAM_WIDTH  * DISPLAY_SCALE)
    display_h  = int(STREAM_HEIGHT * DISPLAY_SCALE)

    t_prev  = time.perf_counter()
    fps_val = 0.0

    print("Window open — press 'q' to quit, 'r' to record, '+'/'-' to tune blur, 'b' for bbox overlay.")

    try:
        while True:
            frames      = pipeline.wait_for_frames(10_000)
            aligned     = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            h, w      = color_bgr.shape[:2]

            # ── Detect ────────────────────────────────────────────────────────
            people = detect_people(color_bgr, seg_model, pose_model, conf_threshold=CONF_THRESHOLD)

            # ── Apply A3 blur to full body (radius from head bbox) ──────────
            anon_frame = color_bgr.copy()
            r_values = []
            for person in people:
                mask = person.mask.astype(bool)
                if person.head_box_xyxy is None:
                    area_frac = box_area(person.bbox_xyxy) / max(h * w, 1)
                    r = a1_body(area_frac, alpha)
                else:
                    r = _a3_radius(person.head_box_xyxy, h, w, alpha)
                
                r_values.append(r)
                if not mask.any():
                    continue
                anon_frame = apply_masked_filter(
                    anon_frame,
                    mask,
                    method="blur",
                    radius=r,
                    depth_value=1.0,   # A3 ignores depth
                    cfg=anon_cfg,
                )

            # Optional bbox overlay on anonymized frame
            if show_body:
                anon_frame = _draw_boxes(anon_frame, people, show_body=True)

            # ── FPS ───────────────────────────────────────────────────────────
            t_now   = time.perf_counter()
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev  = t_now

            # ── Display ───────────────────────────────────────────────────────
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

            cv2.imshow("LA3D Head Anonymization  [RealSense]",
                       cv2.resize(display_frame, (dw, display_h)))

            # ── Record ────────────────────────────────────────────────────────
            if recording:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter("anonymized_head.mp4", fourcc, STREAM_FPS, (w, h))
                writer.write(anon_frame)

            # ── Keys ──────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("Recording saved to anonymized_head.mp4")
                else:
                    print("Recording started …")
            elif key == ord("+") or key == ord("="):
                alpha = min(alpha + 0.1, 5.0)
            elif key == ord("-"):
                alpha = max(alpha - 0.1, 0.1)
            elif key == ord("b"):
                show_body = not show_body
            elif key == ord("d"):
                show_depth = not show_depth

    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
