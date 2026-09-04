#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from common import load_config, output_root, write_csv


def build_rows(cfg):
    rows = []
    upper_n = int(cfg["upper_bound"]["internal_split_train_pairs"])
    for dataset in cfg["datasets"]:
        rows.append({
            "job_id": f"ub_{dataset}",
            "job_type": "upper_bound",
            "dataset": dataset,
            "train_pairs": upper_n,
            "group_size": "",
            "arms": "grid",
        })
    for dataset in cfg["datasets"]:
        for count in cfg["split"]["train_pair_counts"]:
            for group_size in cfg["optimization"]["group_sizes"]:
                rows.append({
                    "job_id": f"abl_{dataset}_n{int(count)}_g{int(group_size)}",
                    "job_type": "ablation",
                    "dataset": dataset,
                    "train_pairs": int(count),
                    "group_size": int(group_size),
                    "arms": "global+cnn",
                })
    if len(rows) != 56 or len({row["job_id"] for row in rows}) != 56:
        raise AssertionError(f"Expected 56 unique jobs, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    rows = build_rows(cfg)
    path = output_root(cfg) / "job_matrix.csv"
    write_csv(path, rows)
    print(json.dumps({
        "path": str(path),
        "jobs": len(rows),
        "upper_bound_jobs": sum(row["job_type"] == "upper_bound" for row in rows),
        "ablation_jobs": sum(row["job_type"] == "ablation" for row in rows),
        "policy_checkpoints": 2 * sum(row["job_type"] == "ablation" for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
