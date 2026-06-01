#!/usr/bin/env python3
"""Evaluate an RF-DETR checkpoint on a COCO validation split.

Supports temporal checkpoints by packing key/reference frames using
``video_id`` and ``frame_id`` fields from the COCO ``images`` section.
"""

from __future__ import annotations

'''
эта утилита запускает проверку rf-detr на coco validation split.
она умеет проверять и обычную модель, и temporal модель, собирая для нее
соседние кадры по video_id/frame_id так же, как это делается при обучении.
'''

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR on a COCO val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--shape", type=int, default=None)
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["checkpoint", "previous", "surrounding", "duplicate"],
        default="checkpoint",
    )
    parser.add_argument("--temporal-ref-frame-step", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def checkpoint_train_config(model: Any) -> dict[str, Any]:
    args = getattr(getattr(model, "model", None), "args", None)
    train_config = getattr(args, "train_config", None)
    return train_config if isinstance(train_config, dict) else {}


def temporal_offsets(num_refs: int, mode: str, step: int) -> list[int]:
    if num_refs <= 0:
        return []
    if mode == "duplicate":
        return [0] * num_refs
    if mode == "previous":
        return [-(idx + 1) * step for idx in range(num_refs)]
    if mode == "surrounding":
        offsets: list[int] = []
        radius = 1
        while len(offsets) < num_refs:
            offsets.extend([-radius * step, radius * step])
            radius += 1
        return offsets[:num_refs]
    raise ValueError(f"Unknown temporal mode: {mode}")


def build_video_index(images: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        groups.setdefault(str(image.get("video_id", image["file_name"])), []).append(image)
    for key, frames in groups.items():
        groups[key] = sorted(frames, key=lambda item: (item.get("frame_id", 0), item["id"]))
    return groups


def select_reference_images(
    image: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    num_refs: int,
    mode: str,
    step: int,
) -> list[dict[str, Any]]:
    if num_refs <= 0:
        return []
    if mode == "duplicate":
        return [image] * num_refs
    key = str(image.get("video_id", image["file_name"]))
    sequence = groups[key]
    pos_by_id = {item["id"]: idx for idx, item in enumerate(sequence)}
    pos = pos_by_id[image["id"]]
    refs = []
    for offset in temporal_offsets(num_refs, mode, step):
        refs.append(sequence[min(max(pos + offset, 0), len(sequence) - 1)])
    return refs


def preprocess(image_path: Path, shape: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)
    tensor = F.resize(tensor, [shape, shape])
    return F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def main() -> None:
    args = parse_args()
    from rfdetr import from_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = from_checkpoint(args.checkpoint)
    model.model.device = device
    model.model.model.to(device)
    model.model.model.eval()

    ann_path = args.dataset_root / "annotations" / "instances_val2017.json"
    image_dir = args.dataset_root / "val2017"
    coco = COCO(str(ann_path))
    images = coco.loadImgs(coco.getImgIds())
    images = sorted(images, key=lambda item: item["id"])
    if args.max_images > 0:
        images = images[: args.max_images]

    num_refs = int(getattr(model.model_config, "temporal_num_ref_frames", 0))
    train_cfg = checkpoint_train_config(model)
    mode = args.temporal_ref_frame_mode
    if mode == "checkpoint":
        mode = str(train_cfg.get("temporal_ref_frame_mode", "previous"))
    step = args.temporal_ref_frame_step
    if step is None:
        step = int(train_cfg.get("temporal_ref_frame_step", 1))
    shape = args.shape or int(model.model.resolution)
    groups = build_video_index(coco.loadImgs(coco.getImgIds()))

    predictions: list[dict[str, Any]] = []
    print(
        f"checkpoint={args.checkpoint} images={len(images)} refs={num_refs} "
        f"mode={mode} step={step} shape={shape}"
    )
    with torch.no_grad():
        for idx, image in enumerate(images):
            clip_images = [image] + select_reference_images(image, groups, num_refs, mode, step)
            tensor = torch.cat(
                [preprocess(image_dir / item["file_name"], shape, device) for item in clip_images],
                dim=0,
            ).unsqueeze(0)
            outputs = model.model.model(tensor)
            target_sizes = torch.tensor([[image["height"], image["width"]]], device=device)
            result = model.model.postprocess(outputs, target_sizes=target_sizes)[0]

            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu()
            boxes = result["boxes"].detach().cpu()
            keep = scores >= args.threshold
            for score, label, box in zip(scores[keep], labels[keep], boxes[keep]):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": int(label),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
            if (idx + 1) % 50 == 0:
                print(f"processed {idx + 1}/{len(images)}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")

    if not predictions:
        raise SystemExit("No predictions above threshold; COCOeval cannot run.")
    coco_dt = coco.loadRes(predictions)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
