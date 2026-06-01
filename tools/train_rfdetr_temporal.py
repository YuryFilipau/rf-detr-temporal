#!/usr/bin/env python3
"""Train temporal RF-DETR on COCO frame sequences.

The temporal model reads the key frame plus reference frames selected from the
same ``video_id`` sequence.  The default mode is ``previous`` so evaluation stays
causal for driving/autopilot use: detections for frame T never use future frames.
"""

from __future__ import annotations

'''
этот скрипт запускает rf-detr с temporal режимом.
он передает модели текущий кадр и выбранные соседние кадры, чтобы temporal
fusion мог сравнить их query и улучшить устойчивость детекции на видео.
'''

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal RF-DETR on COCO frame sequences.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "data" / "dandelion_dataset2_temporal"),
        help="Dataset root with train2017/val2017/annotations containing video_id/frame_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "exps" / "rfdetr_temporal_prev3"),
        help="Training output directory.",
    )
    parser.add_argument(
        "--variant",
        choices=["nano", "small", "medium", "large"],
        default="small",
        help="RF-DETR model size.",
    )
    parser.add_argument("--num-classes", type=int, default=1, help="Number of foreground classes.")
    parser.add_argument("--resolution", type=int, default=640, help="Training/inference resize resolution.")
    parser.add_argument(
        "--temporal-num-ref-frames",
        type=int,
        default=3,
        help="Number of reference frames packed with every key frame.",
    )
    parser.add_argument(
        "--temporal-fusion-layers",
        type=int,
        default=1,
        help="Number of query-fusion layers applied after the RF-DETR decoder.",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument(
        "--temporal-ref-frame-mode",
        choices=["previous", "surrounding", "duplicate"],
        default="previous",
        help="'previous' is causal and recommended for autopilot testing.",
    )
    parser.add_argument(
        "--temporal-ref-frame-step",
        type=int,
        default=1,
        help="Step between selected reference frames.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-drop", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None, help="Optional RF-DETR checkpoint to resume from.")
    parser.add_argument(
        "--pretrain-weights",
        default=None,
        help=(
            "Optional pretrain weights passed to RF-DETR model construction. "
            "Use 'none' to disable model pretrain weights."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--run-test", action="store_true", help="Evaluate test split after training if available.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA to reduce memory use.")
    parser.add_argument(
        "--progress-bar",
        choices=["tqdm", "rich", "none"],
        default="tqdm",
        help="Training progress bar style.",
    )
    return parser.parse_args()


def get_model_class(variant: str):
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    return {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }[variant]


def normalized_pretrain_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "false", "0"}:
        return None
    return value


def main() -> None:
    args = parse_args()
    model_cls = get_model_class(args.variant)

    model_kwargs = {
        "num_classes": args.num_classes,
        "resolution": args.resolution,
        "temporal_num_ref_frames": args.temporal_num_ref_frames,
        "temporal_fusion_layers": args.temporal_fusion_layers,
        "temporal_dropout": args.temporal_dropout,
    }
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = normalized_pretrain_weights(args.pretrain_weights)

    model = model_cls(**model_kwargs)
    model.train(
        dataset_file="coco",
        dataset_dir=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_drop=args.lr_drop,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        use_ema=not args.no_ema,
        run_test=args.run_test,
        progress_bar=None if args.progress_bar == "none" else args.progress_bar,
        temporal_ref_frame_mode=args.temporal_ref_frame_mode,
        temporal_ref_frame_step=args.temporal_ref_frame_step,
        notes={
            "run_type": "rfdetr_temporal",
            "dataset_root": args.dataset_root,
            "temporal": True,
            "temporal_num_ref_frames": args.temporal_num_ref_frames,
            "temporal_fusion_layers": args.temporal_fusion_layers,
            "temporal_ref_frame_mode": args.temporal_ref_frame_mode,
            "temporal_ref_frame_step": args.temporal_ref_frame_step,
        },
    )


if __name__ == "__main__":
    main()
