#!/usr/bin/env python3
"""Run RF-DETR/RF-DETR-temporal on a video and save an annotated video.

This script is intended for qualitative stress testing on hard driving videos.
It supports temporal checkpoints by packing the key frame together with selected
reference frames in the same channel layout used during training.
"""

from __future__ import annotations

'''
эта утилита прогоняет обученный rf-detr по видео или набору кадров.
она нужна для практической проверки: увидеть не только цифры coco, но и
как модель ведет себя на потоке машин во времени.
'''

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF-DETR on a video and draw detections.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="RF-DETR .pth checkpoint.")
    parser.add_argument("--input-video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-video", type=Path, required=True, help="Output annotated video path.")
    parser.add_argument("--predictions-json", type=Path, default=None, help="Optional per-frame predictions JSON.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold for drawing/saving.")
    parser.add_argument("--shape", type=int, default=None, help="Square inference shape. Defaults to model resolution.")
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["checkpoint", "previous", "surrounding", "duplicate"],
        default="checkpoint",
        help="Reference frame mode for temporal checkpoints.",
    )
    parser.add_argument(
        "--temporal-ref-frame-step",
        type=int,
        default=None,
        help="Reference step. Defaults to checkpoint/train config value if available, otherwise 1.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most this many frames. 0 means all.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec for output video.")
    return parser.parse_args()


def read_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, fps


def checkpoint_train_config(model: Any) -> dict[str, Any]:
    args = getattr(getattr(model, "model", None), "args", None)
    train_config = getattr(args, "train_config", None)
    if isinstance(train_config, dict):
        return train_config
    return {}


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


def preprocess_frame(frame_rgb: np.ndarray, shape: int, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(frame_rgb)
    tensor = F.to_tensor(image).to(device)
    tensor = F.resize(tensor, [shape, shape])
    tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor


def build_input_tensor(
    frames: list[np.ndarray],
    frame_idx: int,
    num_refs: int,
    mode: str,
    step: int,
    shape: int,
    device: torch.device,
) -> torch.Tensor:
    selected = [frame_idx]
    for offset in temporal_offsets(num_refs, mode, step):
        selected.append(min(max(frame_idx + offset, 0), len(frames) - 1))
    tensors = [preprocess_frame(frames[idx], shape, device) for idx in selected]
    return torch.cat(tensors, dim=0).unsqueeze(0)


def draw_detections(
    frame_rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(int(label))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 40, 255), 2)
        text = f"{name} {float(score):.2f}"
        cv2.putText(frame_bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 40, 255), 2)
    return frame_bgr


def main() -> None:
    args = parse_args()
    from rfdetr import from_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = from_checkpoint(args.checkpoint)
    model.model.device = device
    model.model.model.to(device)
    model.model.model.eval()

    num_refs = int(getattr(model.model_config, "temporal_num_ref_frames", 0))
    train_cfg = checkpoint_train_config(model)
    mode = args.temporal_ref_frame_mode
    if mode == "checkpoint":
        mode = str(train_cfg.get("temporal_ref_frame_mode", "previous"))
    step = args.temporal_ref_frame_step
    if step is None:
        step = int(train_cfg.get("temporal_ref_frame_step", 1))
    shape = args.shape or int(model.model.resolution)

    frames, fps = read_video(args.input_video, args.max_frames)
    height, width = frames[0].shape[:2]
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {args.output_video}")

    predictions_out: list[dict[str, Any]] = []
    class_names = list(model.class_names) if getattr(model, "class_names", None) else ["car"]

    print(
        f"checkpoint={args.checkpoint} frames={len(frames)} temporal_refs={num_refs} "
        f"mode={mode} step={step} shape={shape} threshold={args.threshold}"
    )

    try:
        for frame_idx, frame_rgb in enumerate(frames):
            input_tensor = build_input_tensor(frames, frame_idx, num_refs, mode, step, shape, device)
            with torch.no_grad():
                outputs = model.model.model(input_tensor)
                target_sizes = torch.tensor([[height, width]], device=device)
                result = model.model.postprocess(outputs, target_sizes=target_sizes)[0]

            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            keep = scores >= args.threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            writer.write(draw_detections(frame_rgb, boxes, scores, labels, class_names))
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [float(v) for v in box]
                predictions_out.append(
                    {
                        "frame_id": frame_idx,
                        "category_id": int(label),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "score": float(score),
                    }
                )
            if (frame_idx + 1) % 50 == 0:
                print(f"processed {frame_idx + 1}/{len(frames)} frames")
    finally:
        writer.release()

    if args.predictions_json is not None:
        args.predictions_json.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_json.write_text(json.dumps(predictions_out, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {args.output_video}")


if __name__ == "__main__":
    main()
