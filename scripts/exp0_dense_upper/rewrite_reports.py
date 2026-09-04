#!/usr/bin/env python3
"""Rewrite EXP0 comparison CSVs with concise requested column names."""

from __future__ import annotations

import argparse

from common import load_config, output_root, read_csv, write_csv


INTERNAL_COLUMNS = (
    "train_dataset", "train_pairs", "group_size", "NGSC_mIoU",
    "Global_optim_mIoU", "Global-NGSC", "CNN_optim_mIoU", "CNN-NGSC", "Upper_Bound",
)
EXTERNAL_COLUMNS = (
    "train_dataset", "train_pairs", "group_size", "eval_dataset", "NGSC_mIoU",
    "Global_optim_mIoU", "Global-NGSC", "CNN_optim_mIoU", "CNN-NGSC", "Upper_Bound",
)


def one(path):
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    root = output_root(cfg)
    report_root = root / "reports"

    internal_path = report_root / "exp0_internal_comparison.csv"
    external_path = report_root / "exp0_external_comparison.csv"
    internal_source = read_csv(internal_path)
    external_source = read_csv(external_path)
    # Preserve the verbose source tables for auditability before replacing them.
    write_csv(report_root / "exp0_internal_comparison_pre_rename.csv", internal_source)
    write_csv(report_root / "exp0_external_comparison_pre_rename.csv", external_source)

    upper_internal = {}
    for dataset in cfg["datasets"]:
        n32 = one(root / "upper_bound_dense_0p1_internal_n32" / dataset / "summary.csv")
        n128 = one(root / "upper_bound_dense_0p1" / dataset / "internal_summary.csv")
        upper_internal[(dataset, 32)] = float(n32["mIoU_percent"])
        upper_internal[(dataset, 128)] = float(n128["mIoU_percent"])

    internal = []
    for row in internal_source:
        dataset, count = row["train_dataset"], int(row["train_pairs"])
        internal.append({
            "train_dataset": dataset,
            "train_pairs": count,
            "group_size": int(row["group_size"]),
            "NGSC_mIoU": float(row["fixed_mIoU_percent"]),
            "Global_optim_mIoU": float(row["global_mIoU_percent"]),
            "Global-NGSC": float(row["global_minus_fixed_percent_point"]),
            "CNN_optim_mIoU": float(row["cnn_mIoU_percent"]),
            "CNN-NGSC": float(row["cnn_minus_fixed_percent_point"]),
            "Upper_Bound": upper_internal[(dataset, count)],
        })
    write_csv(internal_path, internal, fieldnames=INTERNAL_COLUMNS)

    upper_external = {
        dataset: float(one(
            root / "upper_bound_dense_0p1_external_full" / dataset / "summary.csv"
        )["mIoU_percent"])
        for dataset in cfg["datasets"]
    }
    external = []
    for row in external_source:
        train_dataset = row["train_dataset"]
        eval_dataset = row["eval_dataset"]
        if eval_dataset == "macro_average":
            targets = [dataset for dataset in cfg["datasets"] if dataset != train_dataset]
            upper = sum(upper_external[dataset] for dataset in targets) / len(targets)
        else:
            upper = upper_external[eval_dataset]
        external.append({
            "train_dataset": train_dataset,
            "train_pairs": int(row["train_pairs"]),
            "group_size": int(row["group_size"]),
            "eval_dataset": eval_dataset,
            "NGSC_mIoU": float(row["fixed_mIoU_percent"]),
            "Global_optim_mIoU": float(row["global_mIoU_percent"]),
            "Global-NGSC": float(row["global_minus_fixed_percent_point"]),
            "CNN_optim_mIoU": float(row["cnn_mIoU_percent"]),
            "CNN-NGSC": float(row["cnn_minus_fixed_percent_point"]),
            "Upper_Bound": upper,
        })
    write_csv(external_path, external, fieldnames=EXTERNAL_COLUMNS)
    print(internal_path)
    print(external_path)


if __name__ == "__main__":
    main()
