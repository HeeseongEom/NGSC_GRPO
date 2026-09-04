#!/usr/bin/env python3
"""Select exp2 action/reward and state mode using source validation/LODO only."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from common import ROOT, exp2_root, load_exp2_config, write_csv


ACTION_CANDIDATES = (
    ("eta", "soft_iou_youden", "s4_global_eta"),
    ("eta_q", "soft_full", "s4_global_eta_q"),
    ("full4", "soft_iou_youden", "s4_global_full4"),
    ("full5q", "soft_full", "s4_global_full5q"),
)


def checkpoint(cfg, run_name, method, seed=0):
    return exp2_root(cfg) / "runs" / run_name / method / f"seed_{seed}" / "policy_best.pt"


def select_action(cfg):
    rows = []
    for action_set, reward, run_name in ACTION_CANDIDATES:
        values = []
        for method in cfg["methods"]:
            saved = torch.load(checkpoint(cfg, run_name, method), map_location="cpu")
            value = float(saved["best_val_mIoU"])
            values.append(value)
            rows.append({
                "action_set": action_set, "reward": reward, "run_name": run_name,
                "method": method, "best_source_val_mIoU": value,
                "best_source_val_mIoU_percent": 100 * value,
            })
        rows.append({
            "action_set": action_set, "reward": reward, "run_name": run_name,
            "method": "macro_average", "best_source_val_mIoU": statistics.mean(values),
            "best_source_val_mIoU_percent": 100 * statistics.mean(values),
        })
    macro = [row for row in rows if row["method"] == "macro_average"]
    best = max(macro, key=lambda row: row["best_source_val_mIoU"])
    output = {
        "selection_scope": "source_val_only",
        "action_set": best["action_set"], "reward": best["reward"],
        "source_run_name": best["run_name"],
        "macro_source_val_mIoU": best["best_source_val_mIoU"],
    }
    root = exp2_root(cfg) / "selection"
    write_csv(root / "action_selection.csv", rows)
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_action.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


def select_state(cfg):
    selected_action = json.loads(
        (exp2_root(cfg) / "selection" / "selected_action.json").read_text(encoding="utf-8")
    )
    candidates = (
        ("global", "base11", "s6_lodo_global_base11"),
        ("conditional", "base11", "s6_lodo_conditional_base11"),
        ("conditional", "prompt12", "s7_lodo_conditional_prompt12"),
    )
    rows = []
    for kind, state_mode, prefix in candidates:
        values = []
        for domain in cfg["sources"]:
            run_name = f"{prefix}_{domain}"
            for method in cfg["methods"]:
                saved = torch.load(checkpoint(cfg, run_name, method), map_location="cpu")
                value = float(saved["best_val_mIoU"])
                values.append(value)
                rows.append({
                    "kind": kind, "state_mode": state_mode, "heldout_domain": domain,
                    "method": method, "best_lodo_mIoU": value,
                    "best_lodo_mIoU_percent": 100 * value,
                })
        rows.append({
            "kind": kind, "state_mode": state_mode, "heldout_domain": "macro_average",
            "method": "macro_average", "best_lodo_mIoU": statistics.mean(values),
            "best_lodo_mIoU_percent": 100 * statistics.mean(values),
        })
    macros = [row for row in rows if row["method"] == "macro_average"]
    best = max(macros, key=lambda row: row["best_lodo_mIoU"])
    conditional = max(
        [row for row in macros if row["kind"] == "conditional"],
        key=lambda row: row["best_lodo_mIoU"],
    )
    output = {
        **selected_action,
        "selected_kind": best["kind"],
        "selected_state_mode": best["state_mode"],
        "best_lodo_mIoU": best["best_lodo_mIoU"],
        "conditional_state_mode": conditional["state_mode"],
        "conditional_lodo_mIoU": conditional["best_lodo_mIoU"],
    }
    root = exp2_root(cfg) / "selection"
    write_csv(root / "state_selection.csv", rows)
    (root / "selected_model.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("action", "state"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    (select_action if args.mode == "action" else select_state)(cfg)


if __name__ == "__main__":
    main()
