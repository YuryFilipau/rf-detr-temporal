#!/usr/bin/env python3
"""Train plain RF-DETR on the COCO/temporal-frame dataset.

This baseline intentionally ignores ``video_id`` and ``frame_id`` fields if they
exist in the annotations.  Use it as the fair comparison point for temporal
RF-DETR: same frames, same labels, no temporal fusion.
"""

from __future__ import annotations

'''
этот скрипт запускает обычный rf-detr без temporal.
он нужен как честный baseline: тот же датасет, те же классы, те же эпохи,
но модель не получает соседние кадры и работает только по одной картинке.
'''

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_SRC = PROJECT_ROOT / "external" / "rf-detr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline RF-DETR on COCO frames.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "data" / "dandelion_dataset2_temporal"),
        help="Dataset root with train2017/val2017/annotations.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "exps" / "rfdetr_baseline"),
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
        notes={
            "run_type": "rfdetr_baseline",
            "dataset_root": args.dataset_root,
            "temporal": False,
        },
    )


if __name__ == "__main__":
    main()
