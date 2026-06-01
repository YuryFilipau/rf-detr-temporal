#!/usr/bin/env python3
"""Build a combined temporal COCO dataset.

The output merges an existing dandelion COCO dataset with annotated videos from
``data/test``.  Video frames are split by whole videos, extracted to
``train2017``/``val2017``, and COCO annotations are rewritten with unique IDs,
clipped boxes, ``video_id`` and ``frame_id``.
"""

from __future__ import annotations

'''
эта утилита собирает общий temporal датасет из старого dandelion и новых видео.
она режет видео на кадры, переносит аннотации в coco формат, добавляет
video_id/frame_id и делит данные по целым видео, чтобы train и val не смешивались.
'''

import argparse
import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2


FRAME_NUMBER_RE = re.compile(r"(\d+)(?=\.[^.]+$|$)")
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge dandelion_dataset2 with temporal annotated videos.")
    parser.add_argument("--base-root", type=Path, default=Path("data/dandelion_dataset2_temporal"))
    parser.add_argument("--video-root", type=Path, default=Path("data/test"))
    parser.add_argument("--output-root", type=Path, default=Path("data/car_dataset_temporal"))
    parser.add_argument(
        "--val-videos",
        nargs="+",
        default=["cars11", "cars18", "cars13"],
        help="Whole video directories to place into validation split.",
    )
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place existing dandelion images into the output dataset.",
    )
    return parser.parse_args()


def frame_number_from_name(file_name: str, fallback: int) -> int:
    match = FRAME_NUMBER_RE.search(Path(file_name).name)
    return int(match.group(1)) if match else fallback


def sequence_key(file_name: str) -> tuple[str, int | None]:
    stem = Path(file_name).stem
    match = TRAILING_NUMBER_RE.search(stem)
    if match is None:
        return stem, None
    prefix = stem[: match.start(1)].rstrip("_-. ")
    return prefix or stem, int(match.group(1))


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


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def add_base_split(
    base_root: Path,
    output_root: Path,
    split: str,
    next_image_id: int,
    next_annotation_id: int,
    copy_mode: str,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, Any]]:
    ann_path = base_root / "annotations" / f"instances_{split}2017.json"
    image_dir = base_root / f"{split}2017"
    output_image_dir = output_root / f"{split}2017"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id_map: dict[int, int] = {}
    clipped = 0
    dropped = 0

    for image in sorted(data.get("images", []), key=lambda item: item["id"]):
        new_image_id = next_image_id
        next_image_id += 1
        image_id_map[image["id"]] = new_image_id
        out_file_name = f"dandelion_{split}_{image['file_name']}"
        place_file(image_dir / image["file_name"], output_image_dir / out_file_name, copy_mode, overwrite)

        prefix, inferred_frame_id = sequence_key(image["file_name"])
        new_image = {
            "id": new_image_id,
            "file_name": out_file_name,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "video_id": image.get("video_id", f"dandelion_{split}_{prefix}"),
            "frame_id": int(image.get("frame_id", inferred_frame_id or 0)),
            "source_dataset": str(base_root),
            "source_image_id": image["id"],
        }
        images.append(new_image)

        for ann in anns_by_image.get(image["id"], []):
            bbox = clip_bbox(ann["bbox"], new_image["width"], new_image["height"])
            if bbox is None:
                dropped += 1
                continue
            if [float(v) for v in ann["bbox"]] != bbox:
                clipped += 1
            new_ann = deepcopy(ann)
            new_ann["id"] = next_annotation_id
            next_annotation_id += 1
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = 1
            new_ann["bbox"] = bbox
            new_ann["area"] = float(bbox[2] * bbox[3])
            new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
            annotations.append(new_ann)

    manifest = {
        "source": "base",
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "clipped_bboxes": clipped,
        "dropped_empty_bboxes": dropped,
    }
    return images, annotations, next_image_id, next_annotation_id, manifest


def find_video_file(video_dir: Path) -> Path:
    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if len(videos) != 1:
        raise ValueError(f"Expected exactly one video in {video_dir}, found {len(videos)}")
    return videos[0]


def extract_required_frames(video_path: Path, required_frame_numbers: set[int]) -> dict[int, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: dict[int, Any] = {}
    max_required = max(required_frame_numbers) if required_frame_numbers else -1
    idx = 0
    try:
        while idx <= max_required:
            ok, frame = capture.read()
            if not ok:
                break
            if idx in required_frame_numbers:
                frames[idx] = frame
            idx += 1
    finally:
        capture.release()
    missing = sorted(required_frame_numbers - set(frames))
    if missing:
        raise ValueError(f"Could not decode frames from {video_path}: {missing[:10]}")
    return frames


def write_frame(path: Path, frame, image_ext: str, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_ext == "jpg":
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(str(path), frame)


def add_video_dir(
    video_dir: Path,
    output_root: Path,
    split: str,
    next_image_id: int,
    next_annotation_id: int,
    image_ext: str,
    jpg_quality: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, Any]]:
    ann_path = video_dir / "annotations" / "instances_default.json"
    video_path = find_video_file(video_dir)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    source_images = sorted(data.get("images", []), key=lambda item: item["id"])
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    required = {frame_number_from_name(image["file_name"], idx) for idx, image in enumerate(source_images)}
    decoded_frames = extract_required_frames(video_path, required)
    output_image_dir = output_root / f"{split}2017"
    video_id = f"test_{video_dir.name}"
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clipped = 0
    dropped = 0
    empty_frames = 0

    for local_idx, image in enumerate(source_images):
        frame_number = frame_number_from_name(image["file_name"], local_idx)
        out_file_name = f"{video_id}_frame_{frame_number:06d}.{image_ext}"
        out_path = output_image_dir / out_file_name
        if overwrite or not out_path.exists():
            write_frame(out_path, decoded_frames[frame_number], image_ext, jpg_quality)

        new_image_id = next_image_id
        next_image_id += 1
        width = int(image["width"])
        height = int(image["height"])
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
        image_anns = anns_by_image.get(image["id"], [])
        if not image_anns:
            empty_frames += 1
        for ann in image_anns:
            bbox = clip_bbox(ann["bbox"], width, height)
            if bbox is None:
                dropped += 1
                continue
            if [float(v) for v in ann["bbox"]] != bbox:
                clipped += 1
            new_ann = deepcopy(ann)
            new_ann["id"] = next_annotation_id
            next_annotation_id += 1
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = 1
            new_ann["bbox"] = bbox
            new_ann["area"] = float(bbox[2] * bbox[3])
            new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
            annotations.append(new_ann)

    manifest = {
        "source": "video",
        "video": video_dir.name,
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "empty_frames": empty_frames,
        "clipped_bboxes": clipped,
        "dropped_empty_bboxes": dropped,
        "video_file": str(video_path),
    }
    return images, annotations, next_image_id, next_annotation_id, manifest


def save_coco(output_root: Path, split: str, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> None:
    ann_dir = output_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car", "supercategory": ""}],
    }
    (ann_dir / f"instances_{split}2017.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_root} exists. Pass --overwrite to rebuild.")

    val_videos = set(args.val_videos)
    splits = {
        "train": {"images": [], "annotations": [], "next_image_id": 1, "next_annotation_id": 1},
        "val": {"images": [], "annotations": [], "next_image_id": 1, "next_annotation_id": 1},
    }
    manifest: list[dict[str, Any]] = []

    for split in ["train", "val"]:
        images, anns, next_img, next_ann, item = add_base_split(
            base_root=args.base_root,
            output_root=args.output_root,
            split=split,
            next_image_id=splits[split]["next_image_id"],
            next_annotation_id=splits[split]["next_annotation_id"],
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
        )
        splits[split]["images"].extend(images)
        splits[split]["annotations"].extend(anns)
        splits[split]["next_image_id"] = next_img
        splits[split]["next_annotation_id"] = next_ann
        manifest.append(item)

    for video_dir in sorted(path for path in args.video_root.iterdir() if path.is_dir()):
        split = "val" if video_dir.name in val_videos else "train"
        images, anns, next_img, next_ann, item = add_video_dir(
            video_dir=video_dir,
            output_root=args.output_root,
            split=split,
            next_image_id=splits[split]["next_image_id"],
            next_annotation_id=splits[split]["next_annotation_id"],
            image_ext=args.image_ext,
            jpg_quality=args.jpg_quality,
            overwrite=args.overwrite,
        )
        splits[split]["images"].extend(images)
        splits[split]["annotations"].extend(anns)
        splits[split]["next_image_id"] = next_img
        splits[split]["next_annotation_id"] = next_ann
        manifest.append(item)
        print(f"{video_dir.name} -> {split}: images={len(images)} anns={len(anns)} empty={item['empty_frames']}")

    for split in ["train", "val"]:
        save_coco(args.output_root, split, splits[split]["images"], splits[split]["annotations"])
        print(
            f"{split}: images={len(splits[split]['images'])} "
            f"annotations={len(splits[split]['annotations'])}"
        )

    (args.output_root / "build_manifest.json").write_text(
        json.dumps(
            {
                "base_root": str(args.base_root),
                "video_root": str(args.video_root),
                "val_videos": sorted(val_videos),
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output_root}")


if __name__ == "__main__":
    main()
