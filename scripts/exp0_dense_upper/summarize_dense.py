#!/usr/bin/env python3
"""Collect dense upper bounds and append them to the EXP0 internal comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import fingerprint, load_config, output_root, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    root = output_root(cfg)
    current = fingerprint(cfg)
    internal_path = root / "reports" / "exp0_internal_comparison.csv"
    internal = read_csv(internal_path)
    fixed_lookup = {}
    for row in internal:
        if int(row["train_pairs"]) == 128:
            fixed_lookup[row["train_dataset"]] = float(row["fixed_mIoU_percent"])
    dense = []
    for dataset in cfg["datasets"]:
        path = root / "upper_bound_dense_0p1" / dataset / "internal_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_csv(path)
        if len(rows) != 1 or rows[0]["fingerprint"] != current:
            raise RuntimeError(f"Invalid dense summary: {path}")
        row = dict(rows[0])
        for action_name in ("eta", "tau", "gamma", "kappa_sp"):
            row[action_name] = round(float(row[action_name]), 1)
        canonical_fixed = fixed_lookup[dataset]
        row["fixed_mIoU_percent"] = canonical_fixed
        row["dense_minus_fixed_percent_point"] = float(row["mIoU_percent"]) - canonical_fixed
        dense.append(row)
    report = root / "reports" / "exp0_dense_upper_bound_0p1.csv"
    columns = []
    for row in dense:
        for key in row:
            if key not in columns:
                columns.append(key)
    write_csv(report, dense, fieldnames=columns)

    all_results_path = root / "reports" / "exp0_all_results.csv"
    all_results = [
        row for row in read_csv(all_results_path)
        if row.get("setting") != "dense_grid_upper_bound_0p1"
    ]
    all_results.extend(dense)
    all_columns = []
    for row in all_results:
        for key in row:
            if key not in all_columns:
                all_columns.append(key)
    write_csv(all_results_path, all_results, fieldnames=all_columns)

    lookup = {row["dataset"]: row for row in dense}
    output = []
    for row in internal:
        updated = dict(row)
        found = lookup[row["train_dataset"]]
        if int(row["train_pairs"]) == 128:
            updated["dense_0p1_upper_bound_mIoU_percent"] = found["mIoU_percent"]
            updated["dense_0p1_minus_fixed_percent_point"] = found["dense_minus_fixed_percent_point"]
        else:
            updated["dense_0p1_upper_bound_mIoU_percent"] = ""
            updated["dense_0p1_minus_fixed_percent_point"] = ""
        output.append(updated)
    write_csv(internal_path, output)
    status_path = root / "reports" / "dense_upper_status.json"
    status_path.write_text(json.dumps({
        "fingerprint": current,
        "datasets_expected": len(cfg["datasets"]),
        "datasets_found": len(dense),
        "candidates_per_dataset": 153_791,
        "total_candidate_dataset_evaluations": 153_791 * len(dense),
        "resolution": [224, 224],
        "grid_step": 0.1,
        "summary_csv": str(report),
        "all_results_updated": str(all_results_path),
        "internal_comparison_updated": str(internal_path),
    }, indent=2), encoding="utf-8")
    print(status_path)


if __name__ == "__main__":
    main()
