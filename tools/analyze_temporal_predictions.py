#!/usr/bin/env python3
"""Analyze frame-to-frame detection stability for COCO video-style datasets.

Input predictions are standard COCO result JSON files:
``[{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]``.

The script reports metrics that matter for driving video:
miss rate, recall by object size, detection flicker, one-frame holes, and false
positives per frame.  It does not require tracking IDs; it evaluates whether
objects present in each frame are detected consistently over time.
"""

from __future__ import annotations

'''
эта утилита анализирует предсказания на последовательностях кадров.
она помогает смотреть не только map, но и temporal эффекты: пропуски,
скачки уверенности и нестабильность bbox между соседними кадрами.
'''

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal stability of COCO predictions.")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO ground-truth annotation JSON.")
    parser.add_argument(
        "--pred",
        action="append",
        nargs=2,
        metavar=("NAME", "JSON"),
        required=True,
        help="Prediction set name and COCO result JSON. Can be repeated.",
    )
    parser.add_argument("--score-thr", type=float, default=0.3, help="Prediction score threshold.")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for a GT to be detected.")
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = xywh_to_xyxy(left)
    rx1, ry1, rx2, ry2 = xywh_to_xyxy(right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union > 0.0 else 0.0


def area_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def load_ground_truth(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)
    return images, anns_by_image


def load_predictions(path: Path, score_thr: float) -> dict[int, list[dict[str, Any]]]:
    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in json.loads(path.read_text(encoding="utf-8")):
        if float(pred.get("score", 0.0)) >= score_thr:
            preds_by_image[pred["image_id"]].append(pred)
    return preds_by_image


def ordered_videos(images: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, image in images.items():
        video_id = str(image.get("video_id", "no_video_id"))
        groups[video_id].append(image_id)
    for video_id, ids in groups.items():
        groups[video_id] = sorted(ids, key=lambda image_id: (images[image_id].get("frame_id", 0), image_id))
    return dict(sorted(groups.items()))


def evaluate_prediction_set(
    images: dict[int, dict[str, Any]],
    anns_by_image: dict[int, list[dict[str, Any]]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> dict[str, Any]:
    videos = ordered_videos(images)
    gt_total = 0
    gt_detected = 0
    by_size = {name: {"gt": 0, "detected": 0} for name in ["small", "medium", "large"]}
    frame_presence: dict[str, list[bool]] = {}
    frame_gt_presence: dict[str, list[bool]] = {}
    frame_best_scores: list[float] = []
    unmatched_predictions = 0

    for video_id, image_ids in videos.items():
        detected_sequence: list[bool] = []
        gt_sequence: list[bool] = []
        for image_id in image_ids:
            gt_anns = anns_by_image.get(image_id, [])
            preds = preds_by_image.get(image_id, [])
            used_pred_indices: set[int] = set()
            frame_detected = False
            frame_scores: list[float] = []

            for ann in gt_anns:
                gt_total += 1
                bucket = area_bucket(float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])))
                by_size[bucket]["gt"] += 1

                best_iou = 0.0
                best_idx = -1
                best_score = 0.0
                for idx, pred in enumerate(preds):
                    if idx in used_pred_indices:
                        continue
                    if pred.get("category_id") != ann.get("category_id"):
                        continue
                    value = iou_xywh(ann["bbox"], pred["bbox"])
                    if value > best_iou:
                        best_iou = value
                        best_idx = idx
                        best_score = float(pred.get("score", 0.0))

                if best_iou >= iou_thr and best_idx >= 0:
                    used_pred_indices.add(best_idx)
                    gt_detected += 1
                    by_size[bucket]["detected"] += 1
                    frame_detected = True
                    frame_scores.append(best_score)

            unmatched_predictions += max(0, len(preds) - len(used_pred_indices))
            detected_sequence.append(frame_detected)
            gt_sequence.append(bool(gt_anns))
            if frame_scores:
                frame_best_scores.append(max(frame_scores))

        frame_presence[video_id] = detected_sequence
        frame_gt_presence[video_id] = gt_sequence

    gt_frames = sum(sum(seq) for seq in frame_gt_presence.values())
    detected_frames = sum(
        1
        for video_id, detected_seq in frame_presence.items()
        for detected, has_gt in zip(detected_seq, frame_gt_presence[video_id])
        if has_gt and detected
    )
    flicker_transitions = 0
    one_frame_holes = 0
    valid_transitions = 0
    for video_id, detected_seq in frame_presence.items():
        gt_seq = frame_gt_presence[video_id]
        for prev, cur, has_gt_prev, has_gt_cur in zip(
            detected_seq,
            detected_seq[1:],
            gt_seq,
            gt_seq[1:],
        ):
            if has_gt_prev and has_gt_cur:
                valid_transitions += 1
                flicker_transitions += int(prev != cur)
        for left, mid, right, has_left, has_mid, has_right in zip(
            detected_seq,
            detected_seq[1:],
            detected_seq[2:],
            gt_seq,
            gt_seq[1:],
            gt_seq[2:],
        ):
            if has_left and has_mid and has_right and left and not mid and right:
                one_frame_holes += 1

    score_delta = [
        abs(right - left)
        for left, right in zip(frame_best_scores, frame_best_scores[1:])
    ]
    return {
        "gt_objects": gt_total,
        "detected_gt_objects": gt_detected,
        "object_recall": gt_detected / gt_total if gt_total else 0.0,
        "object_miss_rate": 1.0 - gt_detected / gt_total if gt_total else 0.0,
        "gt_frames": gt_frames,
        "detected_gt_frames": detected_frames,
        "frame_recall": detected_frames / gt_frames if gt_frames else 0.0,
        "frame_miss_rate": 1.0 - detected_frames / gt_frames if gt_frames else 0.0,
        "flicker_transitions": flicker_transitions,
        "flicker_per_100_transitions": 100.0 * flicker_transitions / valid_transitions if valid_transitions else 0.0,
        "one_frame_holes": one_frame_holes,
        "false_positives_per_frame": unmatched_predictions / len(images) if images else 0.0,
        "mean_matched_score": mean(frame_best_scores) if frame_best_scores else 0.0,
        "mean_abs_score_delta": mean(score_delta) if score_delta else 0.0,
        "recall_by_size": {
            key: value["detected"] / value["gt"] if value["gt"] else 0.0
            for key, value in by_size.items()
        },
        "gt_by_size": {key: value["gt"] for key, value in by_size.items()},
    }


def print_report(name: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print(f"gt_objects:              {metrics['gt_objects']}")
    print(f"object_recall:           {metrics['object_recall']:.4f}")
    print(f"object_miss_rate:        {metrics['object_miss_rate']:.4f}")
    print(f"frame_recall:            {metrics['frame_recall']:.4f}")
    print(f"frame_miss_rate:         {metrics['frame_miss_rate']:.4f}")
    print(f"flicker_per_100:         {metrics['flicker_per_100_transitions']:.2f}")
    print(f"one_frame_holes:         {metrics['one_frame_holes']}")
    print(f"false_pos_per_frame:     {metrics['false_positives_per_frame']:.4f}")
    print(f"mean_matched_score:      {metrics['mean_matched_score']:.4f}")
    print(f"mean_abs_score_delta:    {metrics['mean_abs_score_delta']:.4f}")
    print(
        "recall_by_size:          "
        f"small={metrics['recall_by_size']['small']:.4f} "
        f"medium={metrics['recall_by_size']['medium']:.4f} "
        f"large={metrics['recall_by_size']['large']:.4f}"
    )
    print(
        "gt_by_size:              "
        f"small={metrics['gt_by_size']['small']} "
        f"medium={metrics['gt_by_size']['medium']} "
        f"large={metrics['gt_by_size']['large']}"
    )


def main() -> None:
    args = parse_args()
    images, anns_by_image = load_ground_truth(args.annotations)
    for name, pred_path in args.pred:
        metrics = evaluate_prediction_set(
            images,
            anns_by_image,
            load_predictions(Path(pred_path), args.score_thr),
            args.iou_thr,
        )
        print_report(name, metrics)


if __name__ == "__main__":
    main()
