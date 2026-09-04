#!/usr/bin/env python3
"""Summarize EXP0_1 while reusing reward-independent EXP0 baselines and upper bounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import fingerprint, load_config, output_root, read_csv, write_csv


def baseline_report(cfg, name: str) -> Path:
    experiment = cfg["artifacts"]["baseline_reports_experiment"]
    return Path(cfg["_root"]) / cfg["experiment"]["output_root"] / experiment / "reports" / name


def collect_evaluations(cfg) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((output_root(cfg) / "evaluations").glob("**/*_summary.csv")):
        rows.extend(read_csv(path))
    return rows


def internal_comparison(cfg, evaluations: list[dict]) -> list[dict]:
    old_rows = read_csv(baseline_report(cfg, "exp0_internal_comparison.csv"))
    old = {
        (row["train_dataset"], int(row["train_pairs"]), int(row["group_size"])): row
        for row in old_rows
    }
    new = {
        (row["train_dataset"], int(row["train_pairs"]), int(row["group_size"]), row["setting"]): row
        for row in evaluations
        if row["split"] == "internal" and row["dataset"] == row["train_dataset"]
    }
    output = []
    for dataset in cfg["datasets"]:
        for count in cfg["split"]["train_pair_counts"]:
            for group in cfg["optimization"]["group_sizes"]:
                key = (dataset, int(count), int(group))
                base = old[key]
                fixed = float(base["NGSC_mIoU"])
                global_value = float(new[(*key, "global")]["mIoU_percent"])
                cnn_value = float(new[(*key, "cnn")]["mIoU_percent"])
                output.append({
                    "train_dataset": dataset,
                    "train_pairs": int(count),
                    "group_size": int(group),
                    "NGSC_mIoU": fixed,
                    "Global_optim_mIoU": global_value,
                    "Global-NGSC": global_value - fixed,
                    "CNN_optim_mIoU": cnn_value,
                    "CNN-NGSC": cnn_value - fixed,
                    "Upper_Bound": float(base["Upper_Bound"]),
                })
    return output


def external_comparison(cfg, evaluations: list[dict]) -> list[dict]:
    old_rows = read_csv(baseline_report(cfg, "exp0_external_comparison.csv"))
    old = {
        (row["train_dataset"], int(row["train_pairs"]), int(row["group_size"]), row["eval_dataset"]): row
        for row in old_rows
    }
    new = {
        (
            row["train_dataset"], int(row["train_pairs"]), int(row["group_size"]),
            row["dataset"], row["setting"],
        ): row
        for row in evaluations
        if row["split"] == "external"
    }
    output = []
    for dataset in cfg["datasets"]:
        eval_datasets = [name for name in cfg["datasets"] if name != dataset] + ["macro_average"]
        for count in cfg["split"]["train_pair_counts"]:
            for group in cfg["optimization"]["group_sizes"]:
                for eval_dataset in eval_datasets:
                    key = (dataset, int(count), int(group), eval_dataset)
                    base = old[key]
                    fixed = float(base["NGSC_mIoU"])
                    global_value = float(new[(*key, "global")]["mIoU_percent"])
                    cnn_value = float(new[(*key, "cnn")]["mIoU_percent"])
                    output.append({
                        "train_dataset": dataset,
                        "train_pairs": int(count),
                        "group_size": int(group),
                        "eval_dataset": eval_dataset,
                        "NGSC_mIoU": fixed,
                        "Global_optim_mIoU": global_value,
                        "Global-NGSC": global_value - fixed,
                        "CNN_optim_mIoU": cnn_value,
                        "CNN-NGSC": cnn_value - fixed,
                        "Upper_Bound": float(base["Upper_Bound"]),
                    })
    return output


def training_diagnostics(cfg) -> list[dict]:
    rows = []
    for path in sorted((output_root(cfg) / "runs").glob("**/training_log.csv")):
        values = read_csv(path)
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        rows.append({
            "dataset": metadata["dataset"],
            "train_pairs": metadata["train_pair_count"],
            "group_size": metadata["group_size"],
            "kind": metadata["kind"],
            "updates": len(values),
            "reward_first": float(values[0]["reward_mean"]),
            "reward_last": float(values[-1]["reward_mean"]),
            "reward_change": float(values[-1]["reward_mean"]) - float(values[0]["reward_mean"]),
            "positive_empty_first": float(values[0]["positive_empty_fraction"]),
            "positive_empty_last": float(values[-1]["positive_empty_fraction"]),
            "final_reference_kl": float(values[-1]["reference_kl"]),
            "max_ratio_deviation": max(float(row["max_abs_ratio_minus_one"]) for row in values),
            "max_clip_fraction": max(float(row["clip_fraction"]) for row in values),
            "all_core_finite": int(all(
                math.isfinite(float(row[key]))
                for row in values
                for key in ("reward_mean", "reward_std", "policy_loss", "reference_kl", "grad_norm")
            )),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    report_root = output_root(cfg) / "reports"
    evaluations = collect_evaluations(cfg)
    internal = internal_comparison(cfg, evaluations)
    external = external_comparison(cfg, evaluations)
    diagnostics = training_diagnostics(cfg)

    columns = []
    for row in evaluations:
        for key in row:
            if key not in columns:
                columns.append(key)
    if evaluations:
        write_csv(report_root / "exp0_1_all_results.csv", evaluations, fieldnames=columns)
    write_csv(report_root / "exp0_1_internal_comparison.csv", internal)
    write_csv(report_root / "exp0_1_external_comparison.csv", external)
    write_csv(report_root / "exp0_1_training_diagnostics.csv", diagnostics)

    status = {
        "fingerprint": fingerprint(cfg),
        "expected_policy_checkpoints": 96,
        "found_policy_checkpoints": len(list((output_root(cfg) / "runs").glob("**/policy_final.pt"))),
        "expected_training_logs": 96,
        "found_training_logs": len(list((output_root(cfg) / "runs").glob("**/training_log.csv"))),
        "expected_evaluation_summaries": 192,
        "found_evaluation_summaries": len(list((output_root(cfg) / "evaluations").glob("**/*_summary.csv"))),
        "expected_internal_rows": 48,
        "found_internal_rows": len(internal),
        "expected_external_rows": 384,
        "found_external_rows": len(external),
        "training_logs_all_core_finite": int(all(row["all_core_finite"] for row in diagnostics)),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
