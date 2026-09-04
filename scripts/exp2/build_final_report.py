#!/usr/bin/env python3
"""Aggregate exp2 final internal/external results and render the mobile snapshot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from common import ROOT, exp2_root, load_exp2_config, read_csv, write_csv


METHODS = ("MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP")
METRICS = ("mIoU", "foreground_mIoU", "background_IoU", "foreground_Dice", "AUROC", "absent_FP_area")


def finite(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_exp2(cfg, state_mode):
    root = exp2_root(cfg)
    grouped = defaultdict(list)
    for method in METHODS:
        for seed in (0, 1, 2):
            runs = {
                "global_beta_grpo": "s8_final_global",
                "conditional_beta_grpo": f"s8_final_conditional_{state_mode}",
            }
            for setting, run in runs.items():
                for split in ("internal", "external"):
                    path = root / "evaluations" / run / method / f"seed_{seed}" / f"{split}_summary.csv"
                    for row in read_csv(path):
                        grouped[(split, method, setting, row["dataset"])].append(row)
    rows = []
    for (split, method, setting, dataset), values in grouped.items():
        row = {
            "experiment": "exp2", "split": "internal_unused_source" if split == "internal" else "external_target",
            "method": method, "setting": setting, "dataset": dataset, "num_seeds": len(values),
        }
        for metric in METRICS:
            numbers = [number for value in values if (number := finite(value.get(metric))) is not None]
            row[f"{metric}_percent"] = 100 * statistics.mean(numbers) if numbers else ""
            row[f"{metric}_seed_std_percent"] = 100 * statistics.stdev(numbers) if len(numbers) > 1 else ""
        rows.append(row)
    return rows


def baseline_rows():
    rows = []
    for row in read_csv(ROOT / "reports" / "exp1" / "exp1_final_results.csv"):
        if row["setting"] not in {"original_ngsc", "core_fixed", "source_static"}:
            continue
        copied = dict(row)
        copied["experiment"] = "exp1_baseline"
        rows.append(copied)
    return rows


def render(path: Path, rows):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    lookup = {
        (row["split"], row["method"], row["setting"]): float(row["mIoU_percent"])
        for row in rows if row["dataset"] == "macro_average"
    }
    cells = []
    for method in METHODS:
        internal = "internal_unused_source"
        external = "external_target"
        values = [
            lookup[(internal, method, "source_static")],
            lookup[(internal, method, "global_beta_grpo")],
            lookup[(internal, method, "conditional_beta_grpo")],
            lookup[(external, method, "source_static")],
            lookup[(external, method, "global_beta_grpo")],
            lookup[(external, method, "conditional_beta_grpo")],
        ]
        cells.append([
            method, *(f"{value:.3f}" for value in values[:3]),
            f"{values[2]-values[0]:+.3f}", *(f"{value:.3f}" for value in values[3:]),
            f"{values[5]-values[3]:+.3f}",
        ])
    columns = ("Method", "Internal\nStatic", "Internal\nGlobal", "Internal\nConditional", "Internal\nCond−Static", "External\nStatic", "External\nGlobal", "External\nConditional", "External\nCond−Static")
    fig, ax = plt.subplots(figsize=(15.4, 4.9), dpi=180)
    ax.axis("off")
    ax.set_title("EXP2 최종 mIoU 요약 (%) — Global/Conditional은 3 seeds 평균", fontsize=15, fontweight="bold", pad=18)
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10.2); table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#183B56"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F4F7FA" if r % 2 else "white")
            if c in (4, 8):
                value = float(cells[r - 1][c])
                cell.set_facecolor("#DFF3E4" if value > 0 else "#FCE2E2")
                cell.set_text_props(fontweight="bold", color="#176B2C" if value > 0 else "#9B1C1C")
    fig.text(0.5, 0.04, "Internal: source train/val 미사용 1,306장 | External: target 4 datasets macro | model selection에는 external 미사용", ha="center", fontsize=10.5, color="#384B5A")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def macro_wide_rows(rows):
    """Build the exact compact table used by the mobile snapshot."""
    lookup = {
        (row["split"], row["method"], row["setting"]): float(row["mIoU_percent"])
        for row in rows if row["dataset"] == "macro_average"
    }
    compact = []
    for method in METHODS:
        internal = "internal_unused_source"
        external = "external_target"
        values = {
            "internal_static_mIoU_percent": lookup[(internal, method, "source_static")],
            "internal_global_mIoU_percent": lookup[(internal, method, "global_beta_grpo")],
            "internal_conditional_mIoU_percent": lookup[(internal, method, "conditional_beta_grpo")],
            "external_static_mIoU_percent": lookup[(external, method, "source_static")],
            "external_global_mIoU_percent": lookup[(external, method, "global_beta_grpo")],
            "external_conditional_mIoU_percent": lookup[(external, method, "conditional_beta_grpo")],
        }
        values["internal_conditional_minus_static_percent_point"] = (
            values["internal_conditional_mIoU_percent"] - values["internal_static_mIoU_percent"]
        )
        values["external_conditional_minus_static_percent_point"] = (
            values["external_conditional_mIoU_percent"] - values["external_static_mIoU_percent"]
        )
        compact.append({"method": method, **{key: round(value, 3) for key, value in values.items()}})
    return compact


def final_training_diagnostics(cfg, state_mode):
    output = []
    root = exp2_root(cfg)
    for run_name, kind in (
        ("s8_final_global", "global"),
        (f"s8_final_conditional_{state_mode}", "conditional"),
    ):
        for method in METHODS:
            for seed in (0, 1, 2):
                directory = root / "runs" / run_name / method / f"seed_{seed}"
                log = read_csv(directory / "training_log.csv")
                metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
                numeric_fields = (
                    "reward_mean", "reward_std", "advantage_zero_fraction", "policy_loss",
                    "clip_fraction", "max_abs_ratio_minus_one", "reference_kl", "grad_norm",
                )
                all_finite = all(finite(row[field]) is not None for row in log for field in numeric_fields)
                output.append({
                    "kind": kind,
                    "method": method,
                    "seed": seed,
                    "updates_completed": len(log),
                    "reward_first_100": statistics.mean(float(row["reward_mean"]) for row in log[:100]),
                    "reward_last_100": statistics.mean(float(row["reward_mean"]) for row in log[-100:]),
                    "best_source_val_mIoU_percent": 100 * float(metadata["best_val_mIoU"]),
                    "best_update": int(metadata["best_update"]),
                    "final_reference_KL": float(log[-1]["reference_kl"]),
                    "max_abs_ratio_minus_one": max(float(row["max_abs_ratio_minus_one"]) for row in log),
                    "max_clip_fraction": max(float(row["clip_fraction"]) for row in log),
                    "mean_zero_advantage_fraction": statistics.mean(
                        float(row["advantage_zero_fraction"]) for row in log
                    ),
                    "all_logged_values_finite": int(all_finite),
                })
    return output


def main():
    cfg = load_exp2_config()
    selection = json.loads((exp2_root(cfg) / "selection" / "selected_model.json").read_text(encoding="utf-8"))
    rows = baseline_rows() + aggregate_exp2(cfg, selection["conditional_state_mode"])
    split_order = {"internal_unused_source": 0, "external_target": 1}
    method_order = {name: idx for idx, name in enumerate(METHODS)}
    setting_order = {name: idx for idx, name in enumerate(("original_ngsc", "core_fixed", "source_static", "global_beta_grpo", "conditional_beta_grpo"))}
    rows.sort(key=lambda row: (split_order[row["split"]], method_order[row["method"]], setting_order[row["setting"]], row["dataset"]))
    report = ROOT / "reports" / "exp2"
    write_csv(report / "exp2_final_results.csv", rows)
    compact = [row for row in rows if row["dataset"] == "macro_average"]
    write_csv(report / "exp2_macro_long.csv", compact)
    write_csv(report / "exp2_macro_summary.csv", macro_wide_rows(rows))
    write_csv(report / "exp2_final_training_diagnostics.csv", final_training_diagnostics(cfg, selection["conditional_state_mode"]))
    render(report / "exp2_macro_summary_mobile.png", rows)
    print(report / "exp2_final_results.csv")
    print(report / "exp2_macro_summary_mobile.png")


if __name__ == "__main__":
    main()
