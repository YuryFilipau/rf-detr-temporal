#!/usr/bin/env python3
"""Build a temporal COCO validation set from annotated videos.

Expected input layout:

    data/test/<video_name>/<video>.mp4
    data/test/<video_name>/annotations/instances_default.json

The source annotations usually contain frame entries such as
``frame_000123.png`` without actual image files.  This script extracts the
corresponding frames from each video, writes a flat ``val2017`` image folder,
and merges annotations into one COCO JSON with ``video_id`` and ``frame_id``.
"""

from __future__ import annotations

'''
эта утилита готовит отдельный temporal validation set из размеченных видео.
она извлекает ровно те кадры, которые описаны в аннотациях, и приводит их
к структуре coco/rf-detr, чтобы можно было проверить модель на новых видео.
'''

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2


FRAME_NUMBER_RE = re.compile(r"(\d+)(?=\.[^.]+$|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create temporal RF-DETR eval dataset from annotated videos.")
    parser.add_argument("--input-root", type=Path, default=Path("data/test"))
    parser.add_argument("--output-root", type=Path, default=Path("data/test_temporal"))
    parser.add_argument(
        "--image-ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Output frame extension. jpg is smaller; png is lossless.",
    )
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output frames/JSON.")
    return parser.parse_args()


def frame_number_from_name(file_name: str, fallback: int) -> int:
    match = FRAME_NUMBER_RE.search(Path(file_name).name)
    return int(match.group(1)) if match else fallback


def clip_bbox(bbox: list[Any], width: int, height: int) -> list[float] | None:
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0.0 or clipped_h <= 0.0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def find_video_file(video_dir: Path) -> Path:
    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if len(videos) != 1:
        raise ValueError(f"Expected exactly one video in {video_dir}, found {len(videos)}: {videos}")
    return videos[0]


def load_frame(video_path: Path, frame_number: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_number} from {video_path}")
        return frame
    finally:
        capture.release()


def extract_required_frames(video_path: Path, required_frame_numbers: set[int]) -> dict[int, Any]:
    if not required_frame_numbers:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: dict[int, Any] = {}
    max_required = max(required_frame_numbers)
    frame_idx = 0
    try:
        while frame_idx <= max_required:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx in required_frame_numbers:
                frames[frame_idx] = frame
            frame_idx += 1
    finally:
        capture.release()

    missing = sorted(required_frame_numbers - set(frames))
    if missing:
        raise ValueError(f"Could not decode required frames from {video_path}: {missing[:10]}")
    return frames


def write_frame(path: Path, frame, image_ext: str, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_ext == "jpg":
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(str(path), frame)


def process_video_dir(
    video_dir: Path,
    output_images_dir: Path,
    image_ext: str,
    jpg_quality: int,
    image_id_start: int,
    annotation_id_start: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, Any]]:
    ann_path = video_dir / "annotations" / "instances_default.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    video_path = find_video_file(video_dir)
    source = json.loads(ann_path.read_text(encoding="utf-8"))

    source_images = sorted(source.get("images", []), key=lambda item: item["id"])
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in source.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    video_id = video_dir.name
    image_id_map: dict[int, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = image_id_start
    next_annotation_id = annotation_id_start
    extracted = 0
    clipped = 0
    dropped_empty = 0
    required_frame_numbers = {
        frame_number_from_name(image["file_name"], local_index)
        for local_index, image in enumerate(source_images)
    }

    cap = cv2.VideoCapture(str(video_path))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    decoded_frames = extract_required_frames(video_path, required_frame_numbers)

    for local_index, image in enumerate(source_images):
        frame_number = frame_number_from_name(image["file_name"], local_index)
        if frame_number >= video_frame_count:
            raise ValueError(
                f"{video_id}: annotation references frame {frame_number}, "
                f"but video has {video_frame_count} frames."
            )

        out_file_name = f"{video_id}_frame_{frame_number:06d}.{image_ext}"
        out_path = output_images_dir / out_file_name
        if overwrite or not out_path.exists():
            frame = decoded_frames[frame_number]
            write_frame(out_path, frame, image_ext, jpg_quality)
            extracted += 1

        new_image_id = next_image_id
        next_image_id += 1
        image_id_map[image["id"]] = new_image_id
        width = int(image.get("width") or video_width)
        height = int(image.get("height") or video_height)
        images.append(
            {
                "id": new_image_id,
                "file_name": out_file_name,
                "width": width,
                "height": height,
                "video_id": video_id,
                "frame_id": frame_number,
                "source_video": str(video_path),
                "source_image_id": image["id"],
            }
        )

        for ann in annotations_by_image.get(image["id"], []):
            bbox = clip_bbox(ann["bbox"], width, height)
            if bbox is None:
                dropped_empty += 1
                continue
            if [float(v) for v in ann["bbox"]] != bbox:
                clipped += 1
            new_ann = deepcopy(ann)
            new_ann["id"] = next_annotation_id
            next_annotation_id += 1
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = int(new_ann.get("category_id", 1))
            new_ann["bbox"] = bbox
            new_ann["area"] = float(bbox[2] * bbox[3])
            new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
            annotations.append(new_ann)

    manifest = {
        "video_id": video_id,
        "video_path": str(video_path),
        "source_annotations": str(ann_path),
        "video_frame_count": video_frame_count,
        "video_size": [video_width, video_height],
        "video_fps": video_fps,
        "annotated_images": len(source_images),
        "output_images": len(images),
        "annotations": len(annotations),
        "extracted_frames": extracted,
        "clipped_bboxes": clipped,
        "dropped_empty_bboxes": dropped_empty,
    }
    return images, annotations, next_image_id, next_annotation_id, manifest


def main() -> None:
    args = parse_args()
    output_images_dir = args.output_root / "val2017"
    output_ann_dir = args.output_root / "annotations"
    output_ann_path = output_ann_dir / "instances_val2017.json"
    manifest_path = args.output_root / "build_manifest.json"

    if output_ann_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_ann_path} exists. Pass --overwrite to rebuild.")

    video_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    all_images: list[dict[str, Any]] = []
    all_annotations: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for video_dir in video_dirs:
        images, annotations, next_image_id, next_annotation_id, manifest = process_video_dir(
            video_dir=video_dir,
            output_images_dir=output_images_dir,
            image_ext=args.image_ext,
            jpg_quality=args.jpg_quality,
            image_id_start=next_image_id,
            annotation_id_start=next_annotation_id,
            overwrite=args.overwrite,
        )
        all_images.extend(images)
        all_annotations.extend(annotations)
        manifests.append(manifest)
        print(
            f"{video_dir.name}: frames={len(images)} anns={len(annotations)} "
            f"clipped={manifest['clipped_bboxes']} dropped={manifest['dropped_empty_bboxes']}"
        )

    coco = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": 1, "name": "car", "supercategory": ""}],
    }
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    output_ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved images: {output_images_dir}")
    print(f"saved annotations: {output_ann_path}")
    print(f"total images={len(all_images)} annotations={len(all_annotations)} videos={len(manifests)}")


if __name__ == "__main__":
    main()
