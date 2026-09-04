#!/usr/bin/env python3
"""Evaluate several exp2 checkpoints in one cache/mask pass."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import (
    Pair, ROOT, _assemble_multiclass, _binary_auroc, _cache_index, _dice, _nanmean,
    _pair_action, _safe_iou, evidence, exp2_root, get_spec, load_class_masks,
    load_exp2_config, load_policy_checkpoint, load_prompt_state, quantile_from_sorted,
    read_csv, refined_patch_scores, upsample_patch_maps, write_csv,
)


def _auroc_job(args):
    return _binary_auroc(*args)


def summarize(cfg, datasets, per_pair):
    summary = []
    for dataset in list(datasets) + ["macro_average"]:
        if dataset == "macro_average":
            bases = summary
            row = {metric: _nanmean(item[metric] for item in bases) for metric in (
                "foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "AUROC", "absent_FP_area"
            )}
            row.update({"dataset": dataset, "num_images": sum(item["num_images"] for item in bases)})
        else:
            rows = [item for item in per_pair if item["dataset"] == dataset]
            foreground = [
                _nanmean(item["iou"] for item in rows if item["class_name"] == class_name)
                for class_name in get_spec(dataset).foreground_classes
            ]
            bg_by_image = {item["image_id"]: item["background_iou"] for item in rows}
            bg = _nanmean(bg_by_image.values())
            valid = [value for value in foreground if np.isfinite(value)]
            row = {
                "dataset": dataset, "foreground_mIoU": _nanmean(foreground), "background_IoU": bg,
                "mIoU": (sum(valid) + bg) / (len(valid) + 1),
                "foreground_Dice": _nanmean(item["dice"] for item in rows),
                "AUROC": _nanmean(item["auroc"] for item in rows),
                "absent_FP_area": _nanmean(item["absent_fp_area"] for item in rows),
                "num_images": len(bg_by_image),
            }
        for metric in ("foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "AUROC", "absent_FP_area"):
            row[f"{metric}_percent"] = 100 * row[metric] if np.isfinite(row[metric]) else float("nan")
        summary.append(row)
    return summary


@torch.inference_mode()
def run(cfg, checkpoints, split, device, force=False, skip_auroc=False, auroc_workers=3):
    run_device = torch.device(device)
    contexts = []
    for checkpoint in checkpoints:
        saved, policy = load_policy_checkpoint(checkpoint, cfg, run_device)
        result_root = exp2_root(cfg) / "evaluations" / saved["run_name"] / saved["method"] / f"seed_{saved['seed']}"
        result_path = result_root / f"{split}_summary.csv"
        contexts.append({
            "saved": saved, "policy": policy, "result_root": result_root, "result_path": result_path,
            "prompt": load_prompt_state(cfg, saved["method"]) if saved["state_mode"] == "prompt12" else {},
            "mean": None if saved["state_mean"] is None else saved["state_mean"].to(run_device),
            "std": None if saved["state_std"] is None else saved["state_std"].to(run_device),
            "null": None if saved["null_evidence"] is None else saved["null_evidence"].to(run_device),
            "pairs": [],
        })
    methods = {context["saved"]["method"] for context in contexts}
    if len(methods) != 1:
        raise ValueError("All checkpoints must use the same dense method")
    method = methods.pop()
    def result_ready(context):
        path = context["result_path"]
        if not path.is_file():
            return False
        if skip_auroc:
            return True
        summary = read_csv(path)
        macro = next((row for row in summary if row["dataset"] == "macro_average"), None)
        return macro is not None and np.isfinite(float(macro["AUROC"]))

    if not force and all(result_ready(context) for context in contexts):
        return [context["result_path"] for context in contexts]

    data_root = ROOT / "data"
    rows = read_csv(_cache_index(cfg, method, split))
    auroc_pool = None if skip_auroc else ThreadPoolExecutor(max_workers=max(1, int(auroc_workers)))
    try:
        for image_index, index_row in enumerate(rows):
            item = torch.load(index_row["cache_path"], map_location="cpu")
            meta = item["meta"]
            masks = load_class_masks(data_root, meta, size=None)
            refinements, thresholds, gates = [], [], []
            for context in contexts:
                current_thresholds, current_gates = [], []
                names = tuple(context["saved"]["action_names"])
                for class_idx, class_name in enumerate(meta["class_names"]):
                    pair = Pair(
                        meta["dataset"], meta["role"], meta["image_id"], meta["image_relpath"], class_name,
                        bool(masks[class_name].any()), item["hat"][class_idx].float(), int(item["seed_idx"][class_idx]),
                        item["base_affinity"][class_idx].float(), item["coords"].float(), item["state"][class_idx].float(),
                        torch.from_numpy(masks[class_name]), tuple(meta["grid_shape"]),
                    )
                    action = _pair_action(
                        pair, context["policy"], context["saved"]["kind"], cfg, names, run_device,
                        context["saved"]["state_mode"], context["prompt"], context["mean"], context["std"]
                    ).detach()
                    refinements.append(refined_patch_scores(pair, action[None], names, False, 1.0)[0])
                    current_thresholds.append(float(action[names.index("eta")]) if "eta" in names else 1.4)
                    allowed = True
                    if "q" in names:
                        q = action[names.index("q")].reshape(1)
                        delta = quantile_from_sorted(context["null"], q)[0]
                        allowed = bool(evidence(pair.hat.to(run_device), float(cfg["reward"]["rho"])) >= delta)
                    current_gates.append(allowed)
                thresholds.append(current_thresholds)
                gates.append(current_gates)
            score_tensor = upsample_patch_maps(torch.stack(refinements), meta["grid_shape"], meta["image_size"])
            score_tensor = score_tensor.float().cpu().numpy()
            class_count = len(meta["class_names"])
            gt_masks = [masks[name] for name in meta["class_names"]]
            auroc_values = np.full((len(contexts), class_count), np.nan, dtype=np.float64)
            if auroc_pool is not None:
                keys, jobs = [], []
                for context_idx in range(len(contexts)):
                    scores = score_tensor[context_idx * class_count:(context_idx + 1) * class_count]
                    for class_idx, gt in enumerate(gt_masks):
                        if bool(gt.any()) and not bool(gt.all()):
                            keys.append((context_idx, class_idx))
                            jobs.append((scores[class_idx], gt))
                for (context_idx, class_idx), value in zip(keys, auroc_pool.map(_auroc_job, jobs)):
                    auroc_values[context_idx, class_idx] = value
            for context_idx, context in enumerate(contexts):
                scores = score_tensor[context_idx * class_count:(context_idx + 1) * class_count]
                current = np.asarray(thresholds[context_idx], dtype=np.float32)
                current[~np.asarray(gates[context_idx])] = np.inf
                predicted = _assemble_multiclass(scores, current)
                pred_any, gt_any = np.logical_or.reduce(predicted), np.logical_or.reduce(gt_masks)
                bg = _safe_iou(~pred_any, ~gt_any)
                for class_idx, class_name in enumerate(meta["class_names"]):
                    pred, gt = predicted[class_idx], gt_masks[class_idx]
                    context["pairs"].append({
                        "dataset": meta["dataset"], "image_id": meta["image_id"], "class_name": class_name,
                        "gt_present": int(gt.any()), "iou": _safe_iou(pred, gt), "dice": _dice(pred, gt),
                        "auroc": float(auroc_values[context_idx, class_idx]),
                        "absent_fp_area": float(pred.mean()) if not gt.any() else float("nan"),
                        "background_iou": bg,
                    })
            if (image_index + 1) % 250 == 0:
                print(f"{method} {split}: {image_index + 1}/{len(rows)}", flush=True)
    finally:
        if auroc_pool is not None:
            auroc_pool.shutdown(wait=True)

    datasets = cfg["sources"] if split == "internal" else cfg["targets"]
    outputs = []
    for context in contexts:
        context["result_root"].mkdir(parents=True, exist_ok=True)
        write_csv(context["result_path"], summarize(cfg, datasets, context["pairs"]))
        write_csv(context["result_root"] / f"{split}_per_pair.csv", context["pairs"])
        outputs.append(context["result_path"])
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("internal", "external"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-auroc", action="store_true")
    parser.add_argument("--auroc-workers", type=int, default=3)
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    for path in run(
        cfg, args.checkpoint, args.split, args.device, args.force, args.skip_auroc, args.auroc_workers
    ):
        print(path)


if __name__ == "__main__":
    main()
