#!/usr/bin/env python3
"""Build the exp1 final result table and a mobile-friendly macro snapshot."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "outputs" / "grpo_ngsc_feasibility_v1"
REPORT = ROOT / "reports" / "exp1"
METHODS = ("MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP")
RUN_ORDER = ("original_ngsc", "core_fixed", "source_static", "conditional_grpo")
METRICS = (
    "mIoU",
    "foreground_mIoU",
    "background_IoU",
    "foreground_Dice",
    "AUROC",
    "absent_FP_area",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def canonical_run(run: str) -> str:
    return "conditional_grpo" if run.startswith("conditional_grpo_seed") else run


def collect() -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for method in METHODS:
        sources = {
            "external_target": EXPERIMENT / method / "results" / "summary.csv",
            "internal_unused_source": EXPERIMENT / "internal_test" / method / "summary.csv",
        }
        for split, path in sources.items():
            for row in read_csv(path):
                grouped[(split, method, canonical_run(row["run"]), row["dataset"])].append(row)

    output: list[dict[str, object]] = []
    for (split, method, setting, dataset), rows in grouped.items():
        record: dict[str, object] = {
            "experiment": "exp1",
            "split": split,
            "method": method,
            "setting": setting,
            "dataset": dataset,
            "num_seeds": len(rows) if setting == "conditional_grpo" else 1,
        }
        for metric in METRICS:
            values = [value for row in rows if (value := finite(row.get(metric))) is not None]
            record[f"{metric}_percent"] = (
                100.0 * statistics.mean(values) if values else ""
            )
            record[f"{metric}_seed_std_percent"] = (
                100.0 * statistics.stdev(values) if len(values) > 1 else ""
            )
        output.append(record)

    split_order = {"internal_unused_source": 0, "external_target": 1}
    method_order = {name: idx for idx, name in enumerate(METHODS)}
    run_order = {name: idx for idx, name in enumerate(RUN_ORDER)}
    output.sort(
        key=lambda row: (
            split_order[str(row["split"])],
            method_order[str(row["method"])],
            run_order[str(row["setting"])],
            str(row["dataset"]),
        )
    )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(row["split"]), str(row["method"]), str(row["setting"])): row
        for row in rows
        if row["dataset"] == "macro_average"
    }
    compact = []
    for method in METHODS:
        row: dict[str, object] = {"method": method}
        for setting in RUN_ORDER:
            short = {
                "original_ngsc": "Original",
                "core_fixed": "Core",
                "source_static": "Static",
                "conditional_grpo": "GRPO",
            }[setting]
            internal = lookup[("internal_unused_source", method, setting)]
            external = lookup[("external_target", method, setting)]
            row[f"Internal_{short}"] = round(float(internal["mIoU_percent"]), 3)
            row[f"External_{short}"] = round(float(external["mIoU_percent"]), 3)
        row["Internal_GRPO-Static"] = round(
            float(lookup[("internal_unused_source", method, "conditional_grpo")]["mIoU_percent"])
            - float(lookup[("internal_unused_source", method, "source_static")]["mIoU_percent"]),
            3,
        )
        row["External_GRPO-Static"] = round(
            float(lookup[("external_target", method, "conditional_grpo")]["mIoU_percent"])
            - float(lookup[("external_target", method, "source_static")]["mIoU_percent"]),
            3,
        )
        compact.append(row)
    return compact


def render_png(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    columns = (
        "Method",
        "Internal\nOriginal",
        "Internal\nStatic",
        "Internal\nGRPO",
        "Internal\nΔ",
        "External\nOriginal",
        "External\nStatic",
        "External\nGRPO",
        "External\nΔ",
    )
    cells = []
    for row in rows:
        cells.append(
            [
                row["method"],
                f'{row["Internal_Original"]:.3f}',
                f'{row["Internal_Static"]:.3f}',
                f'{row["Internal_GRPO"]:.3f}',
                f'{row["Internal_GRPO-Static"]:+.3f}',
                f'{row["External_Original"]:.3f}',
                f'{row["External_Static"]:.3f}',
                f'{row["External_GRPO"]:.3f}',
                f'{row["External_GRPO-Static"]:+.3f}',
            ]
        )

    fig, ax = plt.subplots(figsize=(14.5, 4.9), dpi=180)
    ax.axis("off")
    ax.set_title(
        "EXP1 최종 mIoU 요약 (%) — GRPO는 3 seeds 평균, Δ = GRPO − Source-Static",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.0)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#183B56")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F4F7FA" if row_idx % 2 else "white")
            if col_idx in (4, 8):
                value = float(cells[row_idx - 1][col_idx])
                cell.set_facecolor("#DFF3E4" if value > 0 else "#FCE2E2")
                cell.set_text_props(fontweight="bold", color="#176B2C" if value > 0 else "#9B1C1C")
    fig.text(
        0.5,
        0.04,
        "Internal: source train/val 미사용 1,306장  |  External: Covid-QU-Ex, MedSeg, HAM10000, PH2 macro",
        ha="center",
        fontsize=10.5,
        color="#384B5A",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = collect()
    full_path = REPORT / "exp1_final_results.csv"
    macro_path = REPORT / "exp1_macro_summary.csv"
    image_path = REPORT / "exp1_macro_summary_mobile.png"
    write_csv(full_path, rows)
    compact = macro_table(rows)
    write_csv(macro_path, compact)
    render_png(image_path, compact)
    print(full_path)
    print(macro_path)
    print(image_path)


if __name__ == "__main__":
    main()
