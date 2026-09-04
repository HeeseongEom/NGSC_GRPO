from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import BASELINES, METHODS, experiment_root


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_method(cfg, method: str) -> Path:
    result_root = experiment_root(cfg) / method / "results"
    seeds = [int(seed) for seed in cfg["grpo"]["seeds"]]
    names = ["original_ngsc", "core_fixed", "source_static"] + [
        f"conditional_grpo_seed{seed}" for seed in seeds
    ]
    missing = [result_root / f"{name}.csv" for name in names if not (result_root / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluations:\n" + "\n".join(str(path) for path in missing))
    rows = []
    lookups: Dict[str, Dict[str, dict]] = {}
    for name in names:
        values = {row["dataset"]: row for row in _read_csv(result_root / f"{name}.csv")}
        lookups[name] = values
        for dataset, value in values.items():
            rows.append(
                {
                    "method": method,
                    "run": name,
                    "dataset": dataset,
                    "mIoU": value["mIoU"],
                    "mIoU_percent": value.get("mIoU_percent", 100.0 * float(value["mIoU"])),
                    "foreground_mIoU": value["foreground_mIoU"],
                    "background_IoU": value["background_IoU"],
                    "foreground_Dice": value["foreground_Dice"],
                    "AUROC": value["AUROC"],
                    "absent_FP_area": value["absent_FP_area"],
                }
            )
    static_macro = float(lookups["source_static"]["macro_average"]["mIoU"])
    original_macro = float(lookups["original_ngsc"]["macro_average"]["mIoU"])
    grpo_macros = [float(lookups[f"conditional_grpo_seed{seed}"]["macro_average"]["mIoU"]) for seed in seeds]
    per_target_deltas = []
    for seed in seeds:
        current = lookups[f"conditional_grpo_seed{seed}"]
        per_target_deltas.append(
            [float(current[target]["mIoU"]) - float(lookups["source_static"][target]["mIoU"]) for target in cfg["targets"]]
        )
    action_collapse = False
    for seed in seeds:
        stats_path = experiment_root(cfg) / method / "diagnostics" / f"conditional_grpo_seed{seed}_action_stats.csv"
        stats = _read_csv(stats_path)
        if any(float(row["boundary_hit_rate"]) > 0.95 for row in stats):
            action_collapse = True
    checks = {
        "grpo_minus_static_macro_points": 100.0 * (float(np.mean(grpo_macros)) - static_macro),
        "grpo_minus_original_macro_points": 100.0 * (float(np.mean(grpo_macros)) - original_macro),
        "target_seed_cells_with_delta_ge_minus_0_5_points": int(
            (100.0 * np.asarray(per_target_deltas) >= -0.5).sum()
        ),
        "target_seed_cells_total": int(np.asarray(per_target_deltas).size),
        "all_seed_macro_directions_positive": bool(all(value > static_macro for value in grpo_macros)),
        "boundary_collapse_detected": action_collapse,
    }
    checks["decision"] = (
        "GO"
        if checks["grpo_minus_static_macro_points"] >= 1.0
        and checks["grpo_minus_original_macro_points"] > 0.0
        and all(sum(delta >= -0.005 for delta in row) >= 3 for row in per_target_deltas)
        and not action_collapse
        and checks["all_seed_macro_directions_positive"]
        else "PARTIAL_OR_NO_GO"
    )
    path = result_root / "summary.csv"
    _write_csv(path, rows)
    (result_root / "feasibility_decision.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def summarize_all(cfg) -> Path:
    rows = []
    for method in METHODS:
        path = summarize_method(cfg, method)
        rows.extend(_read_csv(path))
    output = experiment_root(cfg) / "summary_all_methods.csv"
    _write_csv(output, rows)
    return output
