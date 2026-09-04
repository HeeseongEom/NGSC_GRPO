#!/usr/bin/env python3
"""Export final deterministic action values from all EXP0 Global GRPO runs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "exp0"))

from common import ACTION_NAMES, load_config, load_policy, map_actions, run_dir  # noqa: E402


OUTPUT = (
    ROOT / "outputs" / "ngsc_grpo_exp0" / "reports"
    / "exp0_global_grpo_action_values.csv"
)


def main() -> None:
    cfg = load_config()
    rows: list[dict[str, int | str | float]] = []

    for dataset in cfg["datasets"]:
        for train_pairs in cfg["split"]["train_pair_counts"]:
            for group_size in cfg["optimization"]["group_sizes"]:
                checkpoint = run_dir(cfg, dataset, train_pairs, group_size, "global") / "policy_final.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)

                _, policy = load_policy(checkpoint, cfg, torch.device("cpu"))
                with torch.no_grad():
                    alpha, beta = policy.parameters_ab(1)
                    normalized_mean = alpha[0] / (alpha[0] + beta[0])
                    action = map_actions(normalized_mean, cfg)

                row: dict[str, int | str | float] = {
                    "train_dataset": dataset,
                    "train_pairs": int(train_pairs),
                    "group_size": int(group_size),
                }
                row.update({name: round(float(action[i]), 6) for i, name in enumerate(ACTION_NAMES)})
                rows.append(row)

    expected = len(cfg["datasets"]) * len(cfg["split"]["train_pair_counts"]) * len(
        cfg["optimization"]["group_sizes"]
    )
    if len(rows) != expected:
        raise AssertionError((len(rows), expected))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    columns = ["train_dataset", "train_pairs", "group_size", *ACTION_NAMES]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
