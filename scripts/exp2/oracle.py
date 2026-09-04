#!/usr/bin/env python3
"""Dense continuous action oracle on the train/val-unused source internal set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    Pair,
    ROOT,
    _assemble_multiclass,
    _cache_index,
    _dice,
    _nanmean,
    _safe_iou,
    action_column,
    action_names,
    evidence,
    exp2_root,
    load_class_masks,
    load_exp2_config,
    load_source_pairs,
    map_actions,
    null_evidence,
    quantile_from_sorted,
    read_csv,
    refined_patch_scores,
    reward_actions,
    source_static_action,
    upsample_patch_maps,
    write_csv,
    get_spec,
)


def load_internal_pairs(cfg, method: str) -> list[Pair]:
    pairs = []
    data_root = ROOT / "data"
    for row in read_csv(_cache_index(cfg, method, "internal")):
        item = torch.load(row["cache_path"], map_location="cpu")
        meta = item["meta"]
        masks = load_class_masks(data_root, meta, size=(224, 224))
        for idx, class_name in enumerate(meta["class_names"]):
            mask = torch.from_numpy(masks[class_name]).bool()
            pairs.append(
                Pair(
                    dataset=meta["dataset"], role="internal_test", image_id=meta["image_id"],
                    image_relpath=meta["image_relpath"], class_name=class_name,
                    present=bool(mask.any()), hat=item["hat"][idx].float(),
                    seed_idx=int(item["seed_idx"][idx]), base_affinity=item["base_affinity"][idx].float(),
                    coords=item["coords"].float(), state=item["state"][idx].float(), gt_mask=mask,
                    grid_shape=tuple(int(v) for v in meta["grid_shape"]),
                )
            )
    return pairs


def patch_reward(pair, actions, names, reward_name, cfg, null_values):
    settings = cfg["reward"]
    soft = reward_name != "hard"
    score = refined_patch_scores(
        pair, actions, names, soft_affinity=soft,
        temperature_tau=float(settings["temperature_tau"])
    )
    eta = action_column(actions, names, "eta", 1.4)
    gt = F.interpolate(
        pair.gt_mask.float()[None, None], size=pair.grid_shape, mode="nearest"
    )[0, 0].reshape(-1).to(actions.device)
    prediction = (
        torch.sigmoid((score - eta[:, None]) / float(settings["temperature_eta"]))
        if soft else (score >= eta[:, None]).float()
    )
    gate = prediction.new_ones(prediction.shape[0])
    if "q" in names:
        q = action_column(actions, names, "q", 0.95)
        delta = quantile_from_sorted(null_values, q)
        pair_e = evidence(pair.hat.to(actions.device), float(settings["rho"]))
        gate = (
            torch.sigmoid((pair_e - delta) / float(settings["temperature_evidence"]))
            if soft else (pair_e >= delta).float()
        )
        prediction = prediction * gate[:, None]
    if reward_name == "hard":
        if pair.present:
            inter = (prediction * gt).sum(-1)
            return 2 * inter / (prediction.sum(-1) + gt.sum()).clamp_min(1e-8)
        return 1 - prediction.mean(-1)
    if pair.present:
        inter = (prediction * gt).sum(-1)
        union = prediction.sum(-1) + gt.sum() - inter
        iou = inter / union.clamp_min(1e-8)
        tpr = inter / gt.sum().clamp_min(1e-8)
        neg = 1 - gt
        fpr = (prediction * neg).sum(-1) / neg.sum().clamp_min(1e-8)
        pixel = iou if reward_name == "soft_iou" else (
            float(settings["soft_iou_weight"]) * iou
            + float(settings["soft_youden_weight"]) * 0.5 * (tpr - fpr + 1)
        )
    else:
        pixel = 1 - prediction.mean(-1)
    if reward_name != "soft_full":
        return pixel
    target = prediction.new_full(gate.shape, float(pair.present))
    presence = 1 - (gate - target).square()
    return float(settings["pixel_weight_with_presence"]) * pixel + float(settings["presence_weight"]) * presence


def run_oracle(cfg, method, action_set, reward_name, device, force=False):
    names = action_names(cfg, action_set)
    root = exp2_root(cfg) / "oracle" / f"{action_set}_{reward_name}" / method
    result_path = root / "oracle_actions.csv"
    if result_path.is_file() and not force:
        return result_path
    run_device = torch.device(device)
    source = [p for p in load_source_pairs(cfg, method) if p.role == "source_train"]
    internal = load_internal_pairs(cfg, method)
    null_values = null_evidence(source, float(cfg["reward"]["rho"])).to(run_device) if "q" in names else None
    count = int(cfg["oracle"]["candidates"])
    sobol = torch.quasirandom.SobolEngine(
        dimension=len(names), scramble=True, seed=int(cfg["oracle"]["seed"])
    )
    normalized = sobol.draw(count)
    candidates = map_actions(normalized, cfg, names)
    static = source_static_action(cfg, method, names)
    candidates[0] = static
    chunk = int(cfg["oracle"]["candidate_chunk"])
    topk = min(32, count)
    rows = []
    with torch.inference_mode():
        for pair_idx, pair in enumerate(internal):
            approximate = []
            for start in range(0, count, chunk):
                actions = candidates[start:start + chunk].to(run_device)
                approximate.append(patch_reward(pair, actions, names, reward_name, cfg, null_values).cpu())
            approximate = torch.cat(approximate)
            indices = torch.topk(approximate, topk).indices
            refined_actions = candidates[indices].to(run_device)
            exact = reward_actions(pair, refined_actions, names, reward_name, cfg, null_values)
            local = int(exact.argmax())
            best_idx = int(indices[local])
            best_action = candidates[best_idx]
            static_reward = float(reward_actions(pair, static[None].to(run_device), names, reward_name, cfg, null_values)[0])
            row = {
                "dataset": pair.dataset, "image_id": pair.image_id, "class_name": pair.class_name,
                "key": pair.key, "present": int(pair.present), "candidate_index": best_idx,
                "oracle_reward": float(exact[local]), "static_reward": static_reward,
                "reward_gain": float(exact[local]) - static_reward,
            }
            row.update({name: float(best_action[j]) for j, name in enumerate(names)})
            rows.append(row)
            if (pair_idx + 1) % 100 == 0:
                print(json.dumps({"method": method, "pair": pair_idx + 1, "total": len(internal)}), flush=True)
    write_csv(result_path, rows)
    summary = []
    for dataset in list(cfg["sources"]) + ["macro_average"]:
        selected = rows if dataset == "macro_average" else [r for r in rows if r["dataset"] == dataset]
        positive = [r["reward_gain"] for r in selected if r["present"]]
        negative = [r["reward_gain"] for r in selected if not r["present"]]
        summary.append({
            "method": method, "action_set": action_set, "reward": reward_name, "dataset": dataset,
            "num_pairs": len(selected),
            "oracle_reward": float(np.mean([r["oracle_reward"] for r in selected])),
            "static_reward": float(np.mean([r["static_reward"] for r in selected])),
            "reward_gain": float(np.mean([r["reward_gain"] for r in selected])),
            "positive_gain": float(np.mean(positive)) if positive else float("nan"),
            "negative_gain": float(np.mean(negative)) if negative else float("nan"),
        })
    write_csv(root / "reward_summary.csv", summary)
    evaluate_oracle(cfg, method, names, rows, null_values, root, run_device)
    return result_path


def evaluate_oracle(cfg, method, names, action_rows, null_values, root, device):
    lookup = {row["key"]: row for row in action_rows}
    data_root = ROOT / "data"
    per_pair = []
    with torch.inference_mode():
        for index_row in read_csv(_cache_index(cfg, method, "internal")):
            item = torch.load(index_row["cache_path"], map_location="cpu")
            meta = item["meta"]
            masks = load_class_masks(data_root, meta, size=None)
            scores, thresholds, gt_masks = [], [], []
            for idx, class_name in enumerate(meta["class_names"]):
                key = "|".join((meta["dataset"], meta["image_id"], class_name))
                chosen = lookup[key]
                action = torch.tensor([chosen[name] for name in names], device=device)
                pair = Pair(
                    meta["dataset"], "internal_test", meta["image_id"], meta["image_relpath"], class_name,
                    bool(masks[class_name].any()), item["hat"][idx].float(), int(item["seed_idx"][idx]),
                    item["base_affinity"][idx].float(), item["coords"].float(), item["state"][idx].float(),
                    torch.from_numpy(masks[class_name]), tuple(meta["grid_shape"]),
                )
                refined = refined_patch_scores(pair, action[None], names, False, 1.0)[0]
                score = upsample_patch_maps(refined, pair.grid_shape, meta["image_size"])
                eta = float(action[names.index("eta")]) if "eta" in names else 1.4
                allowed = True
                if "q" in names:
                    delta = quantile_from_sorted(null_values, action[names.index("q")][None])[0]
                    allowed = bool(evidence(pair.hat.to(device), float(cfg["reward"]["rho"])) >= delta)
                scores.append(score.cpu().numpy())
                thresholds.append(eta if allowed else float("inf"))
                gt_masks.append(masks[class_name])
            predicted = _assemble_multiclass(np.stack(scores), np.asarray(thresholds, dtype=np.float32))
            pred_any, gt_any = np.logical_or.reduce(predicted), np.logical_or.reduce(gt_masks)
            bg = _safe_iou(~pred_any, ~gt_any)
            for idx, class_name in enumerate(meta["class_names"]):
                per_pair.append({
                    "dataset": meta["dataset"], "image_id": meta["image_id"], "class_name": class_name,
                    "iou": _safe_iou(predicted[idx], gt_masks[idx]),
                    "dice": _dice(predicted[idx], gt_masks[idx]), "background_iou": bg,
                })
    summary = []
    for dataset in cfg["sources"]:
        rows = [r for r in per_pair if r["dataset"] == dataset]
        fg = [_nanmean(r["iou"] for r in rows if r["class_name"] == name) for name in get_spec(dataset).foreground_classes]
        bg = _nanmean({r["image_id"]: r["background_iou"] for r in rows}.values())
        valid = [x for x in fg if np.isfinite(x)]
        summary.append({"method": method, "dataset": dataset, "foreground_mIoU": _nanmean(fg), "background_IoU": bg, "mIoU": (sum(valid) + bg) / (len(valid) + 1)})
    macro = {"method": method, "dataset": "macro_average"}
    for metric in ("foreground_mIoU", "background_IoU", "mIoU"):
        macro[metric] = _nanmean(r[metric] for r in summary)
    summary.append(macro)
    for row in summary:
        for metric in ("foreground_mIoU", "background_IoU", "mIoU"):
            row[f"{metric}_percent"] = 100 * row[metric]
    write_csv(root / "internal_mIoU_summary.csv", summary)
    write_csv(root / "internal_per_pair.csv", per_pair)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--action-set", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    print(run_oracle(cfg, args.method, args.action_set, args.reward, args.device, args.force))


if __name__ == "__main__":
    main()
