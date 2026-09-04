#!/usr/bin/env python3
"""Dataset-level finite-grid upper bound on the n=128 internal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ACTION_NAMES,
    Pair,
    evaluate_fixed,
    evaluate_provider,
    fingerprint,
    fixed_action,
    grid_actions,
    load_config,
    load_text_delta,
    output_root,
    read_csv,
    refined_patch_scores,
    selected_image_rows,
    write_csv,
)


def _candidate_metrics(cfg, rows, actions, device, exact_224: bool):
    run_device = torch.device(device)
    actions = actions.to(run_device)
    action_count = actions.shape[0]
    class_names = None
    foreground_sum = None
    foreground_count = None
    background_sum = torch.zeros(action_count, dtype=torch.float64, device=run_device)
    image_count = 0
    text_delta = None

    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            item = torch.load(row["cache_path"], map_location="cpu")
            if class_names is None:
                class_names = list(item["meta"]["class_names"])
                foreground_sum = torch.zeros(
                    action_count, len(class_names), dtype=torch.float64, device=run_device
                )
                foreground_count = torch.zeros(len(class_names), dtype=torch.float64, device=run_device)
                text_delta = load_text_delta(cfg, item["meta"]["dataset"])
            grid_shape = tuple(int(v) for v in item["meta"]["grid_shape"])
            if exact_224:
                gt = item["gt_masks"].bool().to(run_device)
            else:
                gt = F.interpolate(
                    item["gt_masks"].float().unsqueeze(1), size=grid_shape, mode="nearest"
                )[:, 0].bool().to(run_device)
            gt = gt.reshape(len(class_names), -1)
            score_by_class = []
            for class_idx, class_name in enumerate(class_names):
                pair = Pair(
                    item["meta"]["dataset"], item["meta"]["image_id"], class_name, class_idx,
                    bool(item["present"][class_idx]), item["local"].float(), text_delta[class_idx],
                    item["hat"][class_idx].float(), int(item["seed_idx"][class_idx]),
                    item["base_affinity"][class_idx].float(), item["coords"].float(),
                    item["gt_masks"][class_idx].bool(), grid_shape,
                )
                refined = refined_patch_scores(pair, actions)
                if exact_224:
                    refined = F.interpolate(
                        refined.reshape(action_count, 1, *grid_shape),
                        size=item["gt_masks"].shape[-2:], mode="bilinear", align_corners=False,
                    )[:, 0]
                score_by_class.append(refined.reshape(action_count, -1))
            scores = torch.stack(score_by_class, dim=1)
            eligible = scores >= actions[:, 0, None, None]
            any_eligible = eligible.any(1)
            winner = torch.where(eligible, scores, -torch.inf).argmax(1)
            class_ids = torch.arange(len(class_names), device=run_device)[None, :, None]
            prediction = any_eligible[:, None, :] & (winner[:, None, :] == class_ids)
            for class_idx in range(len(class_names)):
                if bool(gt[class_idx].any()):
                    intersection = (prediction[:, class_idx] & gt[class_idx][None]).sum(-1).double()
                    union = (prediction[:, class_idx] | gt[class_idx][None]).sum(-1).double()
                    foreground_sum[:, class_idx] += intersection / union.clamp_min(1)
                    foreground_count[class_idx] += 1
            pred_any = prediction.any(1)
            gt_any = gt.any(0)
            bg_intersection = ((~pred_any) & (~gt_any)[None]).sum(-1).double()
            bg_union = ((~pred_any) | (~gt_any)[None]).sum(-1).double()
            background_sum += bg_intersection / bg_union.clamp_min(1)
            image_count += 1
            if number % 500 == 0:
                print(json.dumps({
                    "stage": "exact224" if exact_224 else "patch_grid",
                    "images": number, "total": len(rows), "actions": action_count,
                }), flush=True)

    foreground_mean = foreground_sum / foreground_count.clamp_min(1)[None]
    background_mean = background_sum / max(1, image_count)
    miou = torch.cat((foreground_mean, background_mean[:, None]), dim=1).mean(1)
    return {
        "foreground_by_class": foreground_mean.cpu(),
        "background": background_mean.cpu(),
        "miou": miou.cpu(),
        "image_count": image_count,
        "class_names": class_names,
    }


def run(cfg, dataset: str, device: str, force: bool = False) -> Path:
    train_pairs = int(cfg["upper_bound"]["internal_split_train_pairs"])
    root = output_root(cfg) / "upper_bound" / dataset
    final_path = root / "internal_summary.csv"
    metadata_path = root / "metadata.json"
    if final_path.is_file() and metadata_path.is_file() and not force:
        if json.loads(metadata_path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint(cfg):
            return final_path

    rows = selected_image_rows(cfg, dataset, train_pairs, "internal")
    candidates = grid_actions(cfg)
    baseline = fixed_action(cfg, dataset)
    duplicate = torch.isclose(candidates, baseline[None], atol=1e-7).all(1).any()
    if not bool(duplicate):
        candidates = torch.cat((candidates, baseline[None]), dim=0)
    patch = _candidate_metrics(cfg, rows, candidates, device, exact_224=False)
    topk = min(int(cfg["upper_bound"]["exact_topk"]), candidates.shape[0])
    indices = torch.topk(patch["miou"], topk).indices.tolist()
    baseline_index = int(torch.isclose(candidates, baseline[None], atol=1e-7).all(1).nonzero()[0])
    exact_indices = list(dict.fromkeys(indices + [baseline_index]))
    exact_actions = candidates[exact_indices]
    exact = _candidate_metrics(cfg, rows, exact_actions, device, exact_224=True)
    best_local = int(exact["miou"].argmax())
    best_index = exact_indices[best_local]
    best_action = candidates[best_index]

    exact_lookup = {candidate_index: local for local, candidate_index in enumerate(exact_indices)}
    grid_rows = []
    for index, action in enumerate(candidates):
        exact_local = exact_lookup.get(index)
        grid_rows.append({
            "candidate_index": index,
            **{name: float(action[position]) for position, name in enumerate(ACTION_NAMES)},
            "patch_grid_mIoU_percent": 100.0 * float(patch["miou"][index]),
            "exact_224_mIoU_percent": "" if exact_local is None else 100.0 * float(exact["miou"][exact_local]),
            "is_fixed_baseline": int(index == baseline_index),
            "is_selected": int(index == best_index),
        })
    write_csv(root / "grid_results.csv", grid_rows)

    summary, per_pair = evaluate_provider(
        cfg, dataset, train_pairs, "internal", "grid_upper_bound", device,
        kind="constant", constant_action=best_action,
    )
    for row in summary:
        row.update({name: float(best_action[position]) for position, name in enumerate(ACTION_NAMES)})
        row["patch_grid_selection_mIoU_percent"] = 100.0 * float(patch["miou"][best_index])
        row["group_size"] = ""
        row["method"] = cfg["method"]
        row["fingerprint"] = fingerprint(cfg)
    write_csv(final_path, summary)
    if bool(cfg["evaluation"]["save_per_pair"]):
        write_csv(root / "internal_per_pair.csv", per_pair)
    evaluate_fixed(cfg, dataset, train_pairs, "internal", device, force=force)
    metadata_path.write_text(json.dumps({
        "fingerprint": fingerprint(cfg),
        "dataset": dataset,
        "train_pairs_for_internal_split": train_pairs,
        "grid_candidates": int(candidates.shape[0]),
        "exact_224_candidates": len(exact_indices),
        "selected_candidate_index": best_index,
        "selected_action": {name: float(best_action[i]) for i, name in enumerate(ACTION_NAMES)},
        "selection_metric": "dataset multiclass mIoU at 224x224",
        "note": "finite two-stage grid upper bound, not a continuous mathematical upper bound",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    print(run(cfg, args.dataset, args.device, force=args.force))


if __name__ == "__main__":
    main()
