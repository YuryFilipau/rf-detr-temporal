#!/usr/bin/env python3
"""Clip COCO bounding boxes to image boundaries.

This utility fixes annotations where ``bbox=[x, y, w, h]`` extends outside the
image.  It keeps the COCO layout unchanged, updates ``area`` from the clipped
box, and can optionally drop boxes that become empty after clipping.
"""

from __future__ import annotations

'''
эта утилита исправляет bbox в coco аннотациях, если они выходят за границы кадра.
это важно для обучения, потому что некорректные боксы могут ломать loss,
matcher или давать модели неправильный сигнал.
'''

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip COCO bbox coordinates to image boundaries and update area."
    )
    parser.add_argument(
        "annotations",
        nargs="+",
        type=Path,
        help="Path(s) to COCO annotation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Allowed only with one input file. "
            "If omitted, dry-run is used unless --in-place is set."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input JSON after creating a .bak backup.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix used with --in-place. Default: .bak",
    )
    parser.add_argument(
        "--drop-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop annotations that become zero-area after clipping. Default: true.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON with this indentation. Default keeps compact JSON.",
    )
    return parser.parse_args()


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clip_bbox(
    bbox: list[Any],
    image_width: float,
    image_height: float,
) -> tuple[list[float] | None, bool]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")

    x, y, width, height = (float(v) for v in bbox)
    x1 = clip_value(x, 0.0, image_width)
    y1 = clip_value(y, 0.0, image_height)
    x2 = clip_value(x + width, 0.0, image_width)
    y2 = clip_value(y + height, 0.0, image_height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    changed = (
        x1 != x
        or y1 != y
        or clipped_width != width
        or clipped_height != height
    )

    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None, changed

    return [x1, y1, clipped_width, clipped_height], changed


def clean_coco(data: dict[str, Any], drop_empty: bool) -> tuple[dict[str, Any], dict[str, int]]:
    cleaned = deepcopy(data)
    images = cleaned.get("images", [])
    annotations = cleaned.get("annotations", [])
    image_sizes = {
        image["id"]: (float(image["width"]), float(image["height"]))
        for image in images
        if "id" in image and "width" in image and "height" in image
    }

    stats = {
        "annotations": len(annotations),
        "changed_bbox": 0,
        "updated_area": 0,
        "dropped_empty": 0,
        "missing_image": 0,
        "bad_bbox": 0,
    }

    kept_annotations: list[dict[str, Any]] = []
    for annotation in annotations:
        image_id = annotation.get("image_id")
        image_size = image_sizes.get(image_id)
        if image_size is None:
            stats["missing_image"] += 1
            kept_annotations.append(annotation)
            continue

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            stats["bad_bbox"] += 1
            kept_annotations.append(annotation)
            continue

        clipped_bbox, changed = clip_bbox(bbox, *image_size)
        if clipped_bbox is None:
            stats["dropped_empty"] += 1
            if not drop_empty:
                kept_annotations.append(annotation)
            continue

        if changed:
            annotation["bbox"] = clipped_bbox
            stats["changed_bbox"] += 1

        new_area = float(clipped_bbox[2] * clipped_bbox[3])
        old_area = annotation.get("area")
        if old_area is None or abs(float(old_area) - new_area) > 1e-6:
            annotation["area"] = new_area
            stats["updated_area"] += 1

        kept_annotations.append(annotation)

    cleaned["annotations"] = kept_annotations
    return cleaned, stats


def write_json(path: Path, data: dict[str, Any], indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )


def print_stats(path: Path, stats: dict[str, int], output: Path | None) -> None:
    destination = str(output) if output is not None else "dry-run"
    print(f"{path} -> {destination}")
    print(
        "  annotations={annotations} changed_bbox={changed_bbox} "
        "updated_area={updated_area} dropped_empty={dropped_empty} "
        "missing_image={missing_image} bad_bbox={bad_bbox}".format(**stats)
    )


def main() -> None:
    args = parse_args()
    if args.output is not None and len(args.annotations) != 1:
        raise SystemExit("--output can be used only with one annotation file.")
    if args.output is not None and args.in_place:
        raise SystemExit("Use either --output or --in-place, not both.")

    for annotation_path in args.annotations:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        cleaned, stats = clean_coco(data, drop_empty=args.drop_empty)

        output_path: Path | None = None
        if args.in_place:
            backup_path = annotation_path.with_name(annotation_path.name + args.backup_suffix)
            shutil.copy2(annotation_path, backup_path)
            write_json(annotation_path, cleaned, args.indent)
            output_path = annotation_path
            print(f"  backup: {backup_path}")
        elif args.output is not None:
            write_json(args.output, cleaned, args.indent)
            output_path = args.output

        print_stats(annotation_path, stats, output_path)


if __name__ == "__main__":
    main()
