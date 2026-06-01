#!/usr/bin/env python3
"""Build videos and temporal metadata from COCO image sequences.

The script groups frames using COCO ``images`` metadata instead of blindly
joining every file in a split.  This matters for generated driving datasets
where names such as ``train_000123.jpg`` may contain different backgrounds or
different car assets.  By default, grouping uses filename prefix, image size,
frame number gaps, and an optional visual hash break detector.
"""

from __future__ import annotations

'''
эта утилита собирает видео из кадров coco датасета.
она нужна для визуальной проверки последовательностей и аннотаций,
особенно когда мы оцениваем temporal поведение модели.
'''

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - checked at runtime.
    raise SystemExit("Pillow is required: pip install pillow") from exc


TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


@dataclass
class FrameInfo:
    image_id: int
    file_name: str
    width: int
    height: int
    prefix: str
    frame_number: int | None
    path: Path
    annotations: list[dict[str, Any]] = field(default_factory=list)
    hash_bits: int | None = None


@dataclass
class VideoGroup:
    video_id: str
    frames: list[FrameInfo]
    split_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create videos from COCO frames and optionally write video_id/frame_id metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root with train2017/val2017 and annotations/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to process. Default: train val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated_videos"),
        help="Directory for videos and manifests.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--max-videos-per-split",
        type=int,
        default=0,
        help="Encode at most this many videos per split. 0 means no limit.",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=0,
        help=(
            "Resize encoded videos so the longest side is this many pixels. "
            "0 keeps original resolution."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "avi"],
        default="mp4",
        help="Video container. Default: mp4.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2,
        help="Skip video groups with fewer frames. Metadata still appears in the manifest.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=30,
        help=(
            "Split a group when trailing frame numbers jump by more than this value. "
            "Use 0 to disable gap splitting."
        ),
    )
    parser.add_argument(
        "--scene-hash-threshold",
        type=int,
        default=24,
        help=(
            "Split a group when consecutive perceptual frame hashes differ more than "
            "this Hamming distance. Use -1 to disable visual splitting."
        ),
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Average-hash size. 8 means a 64-bit hash.",
    )
    parser.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Draw COCO bounding boxes on output videos for visual annotation checks.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Only write manifest/COCO metadata, do not encode video files.",
    )
    parser.add_argument(
        "--annotated-coco-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for COCO JSON files with added video_id/frame_id fields. "
            "Original annotations are not modified."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grouping stats without writing videos or metadata files.",
    )
    return parser.parse_args()


def split_prefix_and_number(file_name: str) -> tuple[str, int | None]:
    stem = Path(file_name).stem
    match = TRAILING_NUMBER_RE.search(stem)
    if match is None:
        return stem, None
    prefix = stem[: match.start(1)].rstrip("_-. ")
    return prefix or stem, int(match.group(1))


def average_hash(path: Path, hash_size: int) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(gray.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for idx, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << idx
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def load_split(dataset_root: Path, split: str, hash_size: int, use_hash: bool) -> tuple[dict[str, Any], list[FrameInfo]]:
    ann_path = dataset_root / "annotations" / f"instances_{split}2017.json"
    image_dir = dataset_root / f"{split}2017"
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    frames: list[FrameInfo] = []
    missing_files: list[str] = []
    for image in data.get("images", []):
        prefix, frame_number = split_prefix_and_number(image["file_name"])
        path = image_dir / image["file_name"]
        if not path.exists():
            missing_files.append(image["file_name"])
            continue
        frame = FrameInfo(
            image_id=image["id"],
            file_name=image["file_name"],
            width=int(image["width"]),
            height=int(image["height"]),
            prefix=prefix,
            frame_number=frame_number,
            path=path,
            annotations=annotations_by_image.get(image["id"], []),
        )
        if use_hash:
            frame.hash_bits = average_hash(path, hash_size)
        frames.append(frame)

    if missing_files:
        sample = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"{split}: {len(missing_files)} image files are missing, sample: {sample}")

    return data, frames


def frame_sort_key(frame: FrameInfo) -> tuple[int, int | str]:
    if frame.frame_number is None:
        return (1, frame.file_name)
    return (0, frame.frame_number)


def base_group_key(frame: FrameInfo) -> tuple[str, int, int]:
    return (frame.prefix, frame.width, frame.height)


def split_group(
    frames: list[FrameInfo],
    split: str,
    max_frame_gap: int,
    scene_hash_threshold: int,
) -> list[VideoGroup]:
    frames = sorted(frames, key=frame_sort_key)
    groups: list[list[FrameInfo]] = []
    reasons: list[str] = []
    current: list[FrameInfo] = []
    current_reason = "start"

    for frame in frames:
        should_split = False
        reason = "continuous"
        if current:
            prev = current[-1]
            if (
                max_frame_gap > 0
                and prev.frame_number is not None
                and frame.frame_number is not None
                and frame.frame_number - prev.frame_number > max_frame_gap
            ):
                should_split = True
                reason = f"frame_gap>{max_frame_gap}"
            elif (
                scene_hash_threshold >= 0
                and prev.hash_bits is not None
                and frame.hash_bits is not None
                and hamming_distance(prev.hash_bits, frame.hash_bits) > scene_hash_threshold
            ):
                should_split = True
                reason = f"hash_distance>{scene_hash_threshold}"

        if should_split:
            groups.append(current)
            reasons.append(current_reason)
            current = []
            current_reason = reason
        current.append(frame)

    if current:
        groups.append(current)
        reasons.append(current_reason)

    video_groups: list[VideoGroup] = []
    for idx, group_frames in enumerate(groups):
        first = group_frames[0]
        video_id = f"{split}_{first.prefix}_{first.width}x{first.height}_{idx:03d}"
        video_groups.append(VideoGroup(video_id=video_id, frames=group_frames, split_reason=reasons[idx]))
    return video_groups


def build_groups(
    frames: Iterable[FrameInfo],
    split: str,
    max_frame_gap: int,
    scene_hash_threshold: int,
) -> list[VideoGroup]:
    buckets: dict[tuple[str, int, int], list[FrameInfo]] = {}
    for frame in frames:
        buckets.setdefault(base_group_key(frame), []).append(frame)

    groups: list[VideoGroup] = []
    for _, bucket_frames in sorted(buckets.items(), key=lambda item: item[0]):
        groups.extend(split_group(bucket_frames, split, max_frame_gap, scene_hash_threshold))
    return groups


def draw_annotations(image: Image.Image, annotations: list[dict[str, Any]]) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x, y, width, height = annotation.get("bbox", [0, 0, 0, 0])
        x2 = x + width
        y2 = y + height
        draw.rectangle((x, y, x2, y2), outline=(255, 40, 40), width=3)
    return image


def output_size(width: int, height: int, resize_long_edge: int) -> tuple[int, int]:
    if resize_long_edge <= 0 or max(width, height) <= resize_long_edge:
        return width, height
    scale = resize_long_edge / max(width, height)
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    if out_width % 2 == 1:
        out_width += 1
    if out_height % 2 == 1:
        out_height += 1
    return out_width, out_height


def encode_video(
    group: VideoGroup,
    output_path: Path,
    fps: float,
    draw_bboxes: bool,
    resize_long_edge: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = group.frames[0]
    source_size = (first.width, first.height)
    size = output_size(first.width, first.height, resize_long_edge)

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required for video encoding: pip install opencv-python") from exc

    fourcc_name = "mp4v" if output_path.suffix.lower() == ".mp4" else "XVID"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame in group.frames:
            with Image.open(frame.path) as image:
                rgb = image.convert("RGB")
            if draw_bboxes:
                rgb = draw_annotations(rgb, frame.annotations)
            if rgb.size != source_size:
                raise ValueError(f"{frame.file_name} has size {rgb.size}, expected {source_size}")
            if size != source_size:
                rgb = rgb.resize(size, Image.Resampling.BILINEAR)
            bgr = cv2.cvtColor(__import__("numpy").array(rgb), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def group_manifest(group: VideoGroup, video_path: Path | None) -> dict[str, Any]:
    return {
        "video_id": group.video_id,
        "video_path": str(video_path) if video_path is not None else None,
        "split_reason": group.split_reason,
        "num_frames": len(group.frames),
        "width": group.frames[0].width,
        "height": group.frames[0].height,
        "frames": [
            {
                "image_id": frame.image_id,
                "file_name": frame.file_name,
                "frame_number": frame.frame_number,
                "num_annotations": len(frame.annotations),
            }
            for frame in group.frames
        ],
    }


def add_video_fields(data: dict[str, Any], groups: list[VideoGroup]) -> dict[str, Any]:
    video_by_image: dict[int, tuple[str, int]] = {}
    for group in groups:
        for frame_id, frame in enumerate(group.frames):
            video_by_image[frame.image_id] = (group.video_id, frame_id)

    updated = json.loads(json.dumps(data))
    for image in updated.get("images", []):
        video_info = video_by_image.get(image["id"])
        if video_info is None:
            continue
        image["video_id"] = video_info[0]
        image["frame_id"] = video_info[1]
    return updated


def process_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    use_hash = args.scene_hash_threshold >= 0
    data, frames = load_split(args.dataset_root, split, args.hash_size, use_hash)
    groups = build_groups(frames, split, args.max_frame_gap, args.scene_hash_threshold)
    split_output_dir = args.output_dir / split

    manifest_groups: list[dict[str, Any]] = []
    written_videos = 0
    skipped_short = 0
    for group in groups:
        video_path = None
        if len(group.frames) < args.min_frames:
            skipped_short += 1
        elif (
            args.max_videos_per_split > 0
            and written_videos >= args.max_videos_per_split
        ):
            pass
        elif not args.no_videos and not args.dry_run:
            video_path = split_output_dir / f"{group.video_id}.{args.format}"
            encode_video(group, video_path, args.fps, args.draw_bboxes, args.resize_long_edge)
            written_videos += 1
        manifest_groups.append(group_manifest(group, video_path))

    manifest = {
        "split": split,
        "dataset_root": str(args.dataset_root),
        "num_images": len(frames),
        "num_groups": len(groups),
        "written_videos": written_videos,
        "skipped_short_groups": skipped_short,
        "grouping": {
            "base_key": ["filename_prefix", "width", "height"],
            "max_frame_gap": args.max_frame_gap,
            "scene_hash_threshold": args.scene_hash_threshold,
            "hash_size": args.hash_size if use_hash else None,
        },
        "videos": manifest_groups,
    }

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / f"{split}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.annotated_coco_output_dir is not None:
            args.annotated_coco_output_dir.mkdir(parents=True, exist_ok=True)
            updated = add_video_fields(data, groups)
            out_path = args.annotated_coco_output_dir / f"instances_{split}2017.json"
            out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"{split}: images={len(frames)} groups={len(groups)} "
        f"written_videos={written_videos} skipped_short={skipped_short}"
    )
    for item in manifest_groups[:8]:
        print(
            f"  {item['video_id']}: frames={item['num_frames']} "
            f"size={item['width']}x{item['height']} reason={item['split_reason']}"
        )
    if len(manifest_groups) > 8:
        print(f"  ... {len(manifest_groups) - 8} more groups")

    return manifest


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.no_videos = True
    for split in args.splits:
        process_split(args, split)


if __name__ == "__main__":
    main()
