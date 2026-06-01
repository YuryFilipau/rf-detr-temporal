#!/usr/bin/env python3
"""Create readable comparison tables from RF-DETR metrics.csv files."""

from __future__ import annotations

'''
эта утилита приводит логи обучения rf-detr к нормальным таблицам и графикам.
она нужна потому что исходные csv/логи rf-detr неудобно читать напрямую,
а для сравнения baseline и temporal важны одинаковые метрики по эпохам.
'''

import argparse
from pathlib import Path

import pandas as pd


VAL_METRICS = [
    "val/mAP_50_95",
    "val/ema_mAP_50_95",
    "val/mAP_50",
    "val/mAP_75",
    "val/mAR",
    "val/loss",
    "val/precision",
    "val/recall",
    "val/F1",
]

TRAIN_METRICS = [
    "train/loss",
    "train/loss_ce",
    "train/loss_bbox",
    "train/loss_giou",
    "train/class_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize two RF-DETR metrics.csv files into clean tables.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline experiment dir or metrics.csv path.")
    parser.add_argument("--temporal", type=Path, required=True, help="Temporal experiment dir or metrics.csv path.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--temporal-name", default="temporal")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rfdetr_metrics_tables"))
    parser.add_argument("--precision", type=int, default=4)
    return parser.parse_args()


def metrics_path(path: Path) -> Path:
    return path if path.name == "metrics.csv" else path / "metrics.csv"


def load_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics_path(path))
    val = df[df["val/mAP_50_95"].notna()].copy().reset_index(drop=True)
    train = df[df["train/loss"].notna()].copy().reset_index(drop=True)
    return val, train


def best_row(val: pd.DataFrame, metric: str) -> pd.Series:
    idx = val[metric].idxmin() if "loss" in metric else val[metric].idxmax()
    return val.loc[idx]


def build_summary(name: str, val: pd.DataFrame, train: pd.DataFrame) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"run": name}
    final = val.iloc[-1]
    row["final_epoch"] = int(final["epoch"])
    for metric in VAL_METRICS:
        row[f"final_{metric}"] = float(final[metric])
        best = best_row(val, metric)
        row[f"best_{metric}"] = float(best[metric])
        row[f"best_epoch_{metric}"] = int(best["epoch"])
    train_final = train.iloc[-1]
    for metric in TRAIN_METRICS:
        row[f"final_{metric}"] = float(train_final[metric])
    return row


def prefixed_epoch_table(name: str, table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keep = ["epoch"] + metrics
    result = table[keep].copy()
    return result.rename(columns={metric: f"{name}_{metric}" for metric in metrics})


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    val_table: pd.DataFrame,
    train_table: pd.DataFrame,
    deltas: pd.DataFrame,
    precision: int,
) -> None:
    floatfmt = f".{precision}f"
    text = []
    text.append("# RF-DETR Metrics Comparison\n")
    text.append("## Summary\n")
    text.append(summary.to_markdown(index=False, floatfmt=floatfmt))
    text.append("\n\n## Validation By Epoch\n")
    text.append(val_table.to_markdown(index=False, floatfmt=floatfmt))
    text.append("\n\n## Train Loss By Epoch\n")
    text.append(train_table.to_markdown(index=False, floatfmt=floatfmt))
    text.append("\n\n## Temporal Minus Baseline\n")
    text.append(deltas.to_markdown(index=False, floatfmt=floatfmt))
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_val, baseline_train = load_tables(args.baseline)
    temporal_val, temporal_train = load_tables(args.temporal)

    summary = pd.DataFrame(
        [
            build_summary(args.baseline_name, baseline_val, baseline_train),
            build_summary(args.temporal_name, temporal_val, temporal_train),
        ]
    )

    val_table = prefixed_epoch_table(args.baseline_name, baseline_val, VAL_METRICS).merge(
        prefixed_epoch_table(args.temporal_name, temporal_val, VAL_METRICS),
        on="epoch",
        how="outer",
    )
    train_table = prefixed_epoch_table(args.baseline_name, baseline_train, TRAIN_METRICS).merge(
        prefixed_epoch_table(args.temporal_name, temporal_train, TRAIN_METRICS),
        on="epoch",
        how="outer",
    )

    deltas = pd.DataFrame({"epoch": val_table["epoch"]})
    for metric in VAL_METRICS:
        deltas[f"delta_{metric}"] = val_table[f"{args.temporal_name}_{metric}"] - val_table[f"{args.baseline_name}_{metric}"]

    summary.to_csv(args.out_dir / "summary_table.csv", index=False)
    val_table.to_csv(args.out_dir / "validation_by_epoch.csv", index=False)
    train_table.to_csv(args.out_dir / "train_by_epoch.csv", index=False)
    deltas.to_csv(args.out_dir / "temporal_minus_baseline.csv", index=False)
    write_markdown(args.out_dir / "comparison_tables.md", summary, val_table, train_table, deltas, args.precision)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(f"Saved tables to: {args.out_dir}")
    print("\nSUMMARY")
    print(summary.round(args.precision).to_string(index=False))
    print("\nTEMPORAL MINUS BASELINE")
    print(deltas.round(args.precision).to_string(index=False))


if __name__ == "__main__":
    main()
