#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_matrix import build_rows
from common import load_config, output_root, read_csv, write_csv


METRICS = (
    "foreground_mIoU_percent", "background_IoU_percent", "mIoU_percent",
    "foreground_Dice_percent", "absent_FP_area_percent",
)


def collect_results(cfg):
    root = output_root(cfg)
    rows = []
    for path in sorted((root / "evaluations").glob("**/*_summary.csv")):
        rows.extend(read_csv(path))
    for path in sorted((root / "upper_bound").glob("*/internal_summary.csv")):
        rows.extend(read_csv(path))
    return rows


def internal_comparison(cfg, rows):
    output = []
    upper_lookup = {
        row["train_dataset"]: row for row in rows
        if row["setting"] == "grid_upper_bound" and row["dataset"] != "macro_average"
    }
    for dataset in cfg["datasets"]:
        for count in cfg["split"]["train_pair_counts"]:
            fixed = next((row for row in rows if row["setting"] == "fixed_ngsc"
                          and row["split"] == "internal" and row["train_dataset"] == dataset
                          and int(row["train_pairs"]) == int(count) and row["dataset"] == dataset), None)
            for group in cfg["optimization"]["group_sizes"]:
                found = {}
                for kind in ("global", "cnn"):
                    found[kind] = next((row for row in rows if row["setting"] == kind
                                       and row["split"] == "internal" and row["train_dataset"] == dataset
                                       and int(row["train_pairs"]) == int(count)
                                       and int(row["group_size"]) == int(group)
                                       and row["dataset"] == dataset), None)
                if fixed is None and not any(found.values()):
                    continue
                row = {"train_dataset": dataset, "train_pairs": int(count), "group_size": int(group)}
                fixed_value = float(fixed["mIoU_percent"]) if fixed else float("nan")
                row["fixed_mIoU_percent"] = fixed_value
                for kind in ("global", "cnn"):
                    value = float(found[kind]["mIoU_percent"]) if found[kind] else float("nan")
                    row[f"{kind}_mIoU_percent"] = value
                    row[f"{kind}_minus_fixed_percent_point"] = value - fixed_value
                upper = upper_lookup.get(dataset) if int(count) == int(cfg["upper_bound"]["internal_split_train_pairs"]) else None
                row["grid_upper_bound_mIoU_percent"] = float(upper["mIoU_percent"]) if upper else float("nan")
                output.append(row)
    return output


def external_comparison(cfg, rows):
    output = []
    for train_dataset in cfg["datasets"]:
        external_datasets = [name for name in cfg["datasets"] if name != train_dataset]
        for count in cfg["split"]["train_pair_counts"]:
            fixed_by_dataset = {
                row["dataset"]: row for row in rows if row["setting"] == "fixed_ngsc"
                and row["split"] == "external" and row["train_dataset"] == train_dataset
                and int(row["train_pairs"]) == int(count)
            }
            for group in cfg["optimization"]["group_sizes"]:
                for eval_dataset in external_datasets + ["macro_average"]:
                    fixed = fixed_by_dataset.get(eval_dataset)
                    policies = {}
                    for kind in ("global", "cnn"):
                        policies[kind] = next((row for row in rows if row["setting"] == kind
                                               and row["split"] == "external"
                                               and row["train_dataset"] == train_dataset
                                               and int(row["train_pairs"]) == int(count)
                                               and int(row["group_size"]) == int(group)
                                               and row["dataset"] == eval_dataset), None)
                    if fixed is None and not any(policies.values()):
                        continue
                    fixed_value = float(fixed["mIoU_percent"]) if fixed else float("nan")
                    row = {
                        "train_dataset": train_dataset,
                        "train_pairs": int(count),
                        "group_size": int(group),
                        "eval_dataset": eval_dataset,
                        "fixed_mIoU_percent": fixed_value,
                    }
                    for kind in ("global", "cnn"):
                        value = float(policies[kind]["mIoU_percent"]) if policies[kind] else float("nan")
                        row[f"{kind}_mIoU_percent"] = value
                        row[f"{kind}_minus_fixed_percent_point"] = value - fixed_value
                    output.append(row)
    return output


def training_diagnostics(cfg):
    rows = []
    for path in sorted((output_root(cfg) / "runs").glob("**/training_log.csv")):
        values = read_csv(path)
        if not values:
            continue
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        numeric = []
        for row in values:
            numeric.extend(float(row[key]) for key in (
                "reward_mean", "policy_loss", "reference_kl", "grad_norm", "max_abs_ratio_minus_one"
            ))
        rows.append({
            "dataset": metadata["dataset"],
            "train_pairs": metadata["train_pair_count"],
            "group_size": metadata["group_size"],
            "kind": metadata["kind"],
            "updates": len(values),
            "reward_first": float(values[0]["reward_mean"]),
            "reward_last": float(values[-1]["reward_mean"]),
            "reward_change": float(values[-1]["reward_mean"]) - float(values[0]["reward_mean"]),
            "final_reference_kl": float(values[-1]["reference_kl"]),
            "max_ratio_deviation": max(float(row["max_abs_ratio_minus_one"]) for row in values),
            "max_clip_fraction": max(float(row["clip_fraction"]) for row in values),
            "all_finite": int(bool(np.isfinite(numeric).all())),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    report_root = output_root(cfg) / "reports"
    rows = collect_results(cfg)
    if rows:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        write_csv(report_root / "exp0_all_results.csv", rows, fieldnames=columns)
    internal = internal_comparison(cfg, rows)
    external = external_comparison(cfg, rows)
    diagnostics = training_diagnostics(cfg)
    if internal:
        write_csv(report_root / "exp0_internal_comparison.csv", internal)
    if external:
        write_csv(report_root / "exp0_external_comparison.csv", external)
    if diagnostics:
        write_csv(report_root / "exp0_training_diagnostics.csv", diagnostics)
    expected_jobs = build_rows(cfg)
    status = {
        "expected_jobs": len(expected_jobs),
        "expected_upper_bound_jobs": 8,
        "expected_ablation_jobs": 48,
        "expected_policy_checkpoints": 96,
        "found_policy_checkpoints": len(list((output_root(cfg) / "runs").glob("**/policy_final.pt"))),
        "found_upper_bound_summaries": len(list((output_root(cfg) / "upper_bound").glob("*/internal_summary.csv"))),
        "found_internal_comparison_rows": len(internal),
        "found_external_comparison_rows": len(external),
        "found_training_diagnostics": len(diagnostics),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
