"""Vendored from ``src/depth_anon/detection.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class DetectedPerson:
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    score: float
    head_box_xyxy: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectedAccessory:
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    score: float
    class_id: int


def box_area(box_xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box_xyxy
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def clip_box(
    box_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return (x1, y1, x2, y2)


def box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def box_to_mask(
    box_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    x1, y1, x2, y2 = box_xyxy
    mask[y1:y2, x1:x2] = True
    return mask


def derive_head_box_from_keypoints(
    keypoints: np.ndarray,
    body_box_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    _HEAD_KP_INDICES = [0, 1, 2, 3, 4]

    if keypoints.shape[0] > max(_HEAD_KP_INDICES):
        head_kps = keypoints[_HEAD_KP_INDICES]
    else:
        head_kps = keypoints

    if head_kps.shape[1] >= 3:
        visible = head_kps[head_kps[:, 2] > 0.5][:, :2]
    else:
        visible = head_kps[(head_kps[:, :2] > 0).all(axis=1)][:, :2]
    if visible.shape[0] >= 3:
        x_min = int(np.floor(visible[:, 0].min()))
        y_min = int(np.floor(visible[:, 1].min()))
        x_max = int(np.ceil(visible[:, 0].max()))
        y_max = int(np.ceil(visible[:, 1].max()))
        width = max(12, x_max - x_min)
        height = max(12, y_max - y_min)

        face_size = max(width, height)

        pad_x = int(round(face_size * 0.40))
        pad_top = int(round(face_size * 0.85))
        pad_bottom = int(round(face_size * 0.65))

        box = (x_min - pad_x, y_min - pad_top, x_max + pad_x, y_max + pad_bottom)
        height_img, width_img = image_shape
        return clip_box(box, width=width_img, height=height_img)

    return None


def _extract_segmentation_masks(result, image_shape: tuple[int, int]) -> list[np.ndarray]:
    if result.masks is None or result.masks.data is None:
        return []

    mask_tensor = result.masks.data
    if hasattr(mask_tensor, "detach"):
        mask_array = mask_tensor.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_tensor)

    target_h, target_w = image_shape
    masks: list[np.ndarray] = []

    for mask in mask_array:
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        masks.append(mask > 0.5)

    return masks


def _extract_boxes(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    return boxes, scores, classes


def detect_people(
    frame_bgr: np.ndarray,
    seg_model,
    pose_model,
    conf_threshold: float = 0.25,
) -> list[DetectedPerson]:
    image_height, image_width = frame_bgr.shape[:2]
    seg_result = seg_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=conf_threshold,
        classes=[0],
    )[0]
    pose_result = pose_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=conf_threshold,
        classes=[0],
    )[0]

    seg_boxes, seg_scores, seg_classes = _extract_boxes(seg_result)
    seg_masks = _extract_segmentation_masks(seg_result, frame_bgr.shape[:2])
    people: list[DetectedPerson] = []
    for idx, (box, score, cls_id) in enumerate(zip(seg_boxes, seg_scores, seg_classes, strict=False)):
        if int(cls_id) != 0:
            continue
        bbox_xyxy = clip_box(tuple(int(round(v)) for v in box.tolist()), image_width, image_height)
        mask = seg_masks[idx] if idx < len(seg_masks) else box_to_mask(bbox_xyxy, frame_bgr.shape[:2])
        people.append(DetectedPerson(bbox_xyxy=bbox_xyxy, mask=mask, score=float(score)))

    if len(people) == 0:
        return []

    if pose_result.boxes is None or pose_result.keypoints is None:
        return people

    pose_boxes, _, pose_classes = _extract_boxes(pose_result)
    keypoints = pose_result.keypoints.data.detach().cpu().numpy()
    for pose_box, cls_id, keypoint_set in zip(pose_boxes, pose_classes, keypoints, strict=False):
        if int(cls_id) != 0:
            continue
        pose_box_xyxy = clip_box(
            tuple(int(round(v)) for v in pose_box.tolist()),
            image_width,
            image_height,
        )
        best_idx = None
        best_iou = 0.0
        for idx, person in enumerate(people):
            overlap = box_iou(person.bbox_xyxy, pose_box_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou > 0.1:
            people[best_idx].head_box_xyxy = derive_head_box_from_keypoints(
                keypoint_set,
                people[best_idx].bbox_xyxy,
                frame_bgr.shape[:2],
            )

    return people


def detect_people_and_accessories(
    frame_bgr: np.ndarray,
    seg_model,
    pose_model,
    conf_threshold: float = 0.25,
) -> tuple[list[DetectedPerson], list[DetectedAccessory]]:
    image_height, image_width = frame_bgr.shape[:2]
    target_classes = [0, 24, 25, 26, 27, 28, 67]
    seg_result = seg_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=conf_threshold,
        classes=target_classes,
    )[0]
    pose_result = pose_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=conf_threshold,
        classes=[0],
    )[0]

    seg_boxes, seg_scores, seg_classes = _extract_boxes(seg_result)
    seg_masks = _extract_segmentation_masks(seg_result, frame_bgr.shape[:2])
    people: list[DetectedPerson] = []
    accessories: list[DetectedAccessory] = []

    for idx, (box, score, cls_id) in enumerate(zip(seg_boxes, seg_scores, seg_classes, strict=False)):
        bbox_xyxy = clip_box(tuple(int(round(v)) for v in box.tolist()), image_width, image_height)
        mask = seg_masks[idx] if idx < len(seg_masks) else box_to_mask(bbox_xyxy, frame_bgr.shape[:2])
        if int(cls_id) == 0:
            people.append(DetectedPerson(bbox_xyxy=bbox_xyxy, mask=mask, score=float(score)))
        else:
            accessories.append(DetectedAccessory(bbox_xyxy=bbox_xyxy, mask=mask, score=float(score), class_id=int(cls_id)))

    if len(people) == 0 or pose_result.boxes is None or pose_result.keypoints is None:
        return people, accessories

    pose_boxes, _, pose_classes = _extract_boxes(pose_result)
    keypoints = pose_result.keypoints.data.detach().cpu().numpy()
    for pose_box, cls_id, keypoint_set in zip(pose_boxes, pose_classes, keypoints, strict=False):
        if int(cls_id) != 0:
            continue
        pose_box_xyxy = clip_box(
            tuple(int(round(v)) for v in pose_box.tolist()),
            image_width,
            image_height,
        )
        best_idx = None
        best_iou = 0.0
        for idx, person in enumerate(people):
            overlap = box_iou(person.bbox_xyxy, pose_box_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou > 0.1:
            if people[best_idx].head_box_xyxy is None:
                head_box = derive_head_box_from_keypoints(
                    keypoint_set,
                    people[best_idx].bbox_xyxy,
                    frame_bgr.shape[:2],
                )
                if head_box is not None:
                    people[best_idx].head_box_xyxy = head_box

    return people, accessories


def crop_boxes_from_frame(
    frame_bgr: np.ndarray,
    boxes_xyxy: Iterable[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size != 0:
            crops.append(crop)
    return crops
