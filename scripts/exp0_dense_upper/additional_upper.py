#!/usr/bin/env python3
"""Dense 0.1 upper bounds for EXP0 n=32 internal and full target datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import torch

from common import (
    ACTION_NAMES,
    cache_index,
    fingerprint,
    fixed_action,
    load_config,
    output_root,
    read_csv,
    selected_image_rows,
)
from dense_upper import _atomic_csv, evaluate_exact_dense
from ngsc_grpo.registry import get_spec


SCOPES = ("internal_n32", "external_full")


def scope_root(cfg: Mapping, scope: str, dataset: str) -> Path:
    if scope not in SCOPES:
        raise ValueError(scope)
    return output_root(cfg) / f"upper_bound_dense_0p1_{scope}" / dataset


def scope_rows(cfg: Mapping, scope: str, dataset: str) -> list[dict]:
    if scope == "internal_n32":
        return selected_image_rows(cfg, dataset, 32, "internal")
    if scope == "external_full":
        return read_csv(cache_index(cfg, dataset))
    raise ValueError(scope)


def write_outputs(cfg: Mapping, scope: str, dataset: str, result: Mapping) -> Path:
    root = scope_root(cfg, scope, dataset)
    summary_path = root / "summary.csv"
    metadata_path = root / "metadata.json"
    current = fingerprint(cfg)
    etas, spatial, miou = result["etas"], result["spatial"], result["miou"]
    best_spatial, best_eta = int(result["best_spatial"]), int(result["best_eta"])
    tau, gamma, kappa = (round(float(v), 1) for v in spatial[best_spatial])
    eta = round(float(etas[best_eta]), 1)
    foreground = result["foreground_mean"][best_spatial, best_eta]
    finite_foreground = foreground[torch.isfinite(foreground)]
    background = float(result["background_mean"][best_spatial, best_eta])
    best_miou = float(miou[best_spatial, best_eta])
    fixed = fixed_action(cfg, dataset).cpu()
    fixed_eta = int(torch.isclose(etas, fixed[0], atol=1e-6).nonzero()[0])
    fixed_spatial = int(torch.isclose(
        spatial, torch.tensor([fixed[1], fixed[2], fixed[3]])[None], atol=1e-6
    ).all(1).nonzero()[0])
    fixed_miou = float(miou[fixed_spatial, fixed_eta])

    grid_rows = []
    candidate_index = 0
    for spatial_index, values in enumerate(spatial):
        tau_value, gamma_value, kappa_value = (round(float(v), 1) for v in values)
        for eta_index, eta_value in enumerate(etas):
            grid_rows.append({
                "candidate_index": candidate_index,
                "eta": round(float(eta_value), 1),
                "tau": tau_value,
                "gamma": gamma_value,
                "kappa_sp": kappa_value,
                "exact_224_mIoU_percent": 100.0 * float(miou[spatial_index, eta_index]),
                "is_fixed_baseline": int(spatial_index == fixed_spatial and eta_index == fixed_eta),
                "is_selected": int(spatial_index == best_spatial and eta_index == best_eta),
            })
            candidate_index += 1
    _atomic_csv(root / "dense_grid_results.csv", grid_rows, list(grid_rows[0]))

    summary = {
        "dataset": dataset,
        "scope": scope,
        "num_images": int(result["image_count"]),
        "foreground_mIoU_percent": 100.0 * float(finite_foreground.mean()),
        "background_IoU_percent": 100.0 * background,
        "mIoU_percent": 100.0 * best_miou,
        "fixed_grid_mIoU_percent": 100.0 * fixed_miou,
        "upper_minus_fixed_percent_point": 100.0 * (best_miou - fixed_miou),
        "eta": eta,
        "tau": tau,
        "gamma": gamma,
        "kappa_sp": kappa,
        "train_pairs": 32 if scope == "internal_n32" else "",
        "split": "internal" if scope == "internal_n32" else "external_target_oracle",
        "setting": "dense_grid_upper_bound_0p1",
        "method": cfg["method"],
        "fingerprint": current,
    }
    for index, name in enumerate(get_spec(dataset).foreground_classes):
        summary[f"class_{index}_name"] = name
        summary[f"class_{index}_IoU_percent"] = 100.0 * float(foreground[index])
    _atomic_csv(summary_path, [summary], list(summary))
    metadata_path.write_text(json.dumps({
        "fingerprint": current,
        "dataset": dataset,
        "scope": scope,
        "grid_step": 0.1,
        "total_candidates": 153_791,
        "evaluation_resolution": [224, 224],
        "selection_metric": "exact final multiclass mIoU at 224x224",
        "selected_action": dict(zip(ACTION_NAMES, (eta, tau, gamma, kappa))),
        "selected_mIoU_percent": 100.0 * best_miou,
        "candidate_chunk_final": result["candidate_chunk"],
        "image_batch": result["image_batch"],
        "max_gpu_GiB": result["max_gpu_GiB"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def run(cfg: Mapping, scope: str, dataset: str, device: str, force: bool = False) -> Path:
    root = scope_root(cfg, scope, dataset)
    path = root / "summary.csv"
    metadata_path = root / "metadata.json"
    if path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint(cfg):
            return path
    result = evaluate_exact_dense(cfg, dataset, device, rows=scope_rows(cfg, scope, dataset))
    return write_outputs(cfg, scope, dataset, result)


def run_shard(
    cfg: Mapping, scope: str, dataset: str, device: str,
    shard_index: int, num_shards: int, force: bool = False,
) -> Path:
    root = scope_root(cfg, scope, dataset) / "shards"
    path = root / f"part_{shard_index}_of_{num_shards}.pt"
    current = fingerprint(cfg)
    if path.is_file() and not force:
        saved = torch.load(path, map_location="cpu")
        if saved.get("fingerprint") == current:
            return path
    rows = scope_rows(cfg, scope, dataset)[shard_index::num_shards]
    result = evaluate_exact_dense(cfg, dataset, device, rows=rows)
    payload = {
        "fingerprint": current,
        "scope": scope,
        "dataset": dataset,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "etas": result["etas"],
        "spatial": result["spatial"],
        "foreground_sum": result["foreground_sum"],
        "background_sum": result["background_sum"],
        "present_count": result["present_count"],
        "image_count": result["image_count"],
        "candidate_chunk": result["candidate_chunk"],
        "image_batch": result["image_batch"],
        "max_gpu_GiB": result["max_gpu_GiB"],
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def merge_shards(cfg: Mapping, scope: str, dataset: str, num_shards: int) -> Path:
    root = scope_root(cfg, scope, dataset) / "shards"
    parts = [torch.load(root / f"part_{i}_of_{num_shards}.pt", map_location="cpu")
             for i in range(num_shards)]
    current = fingerprint(cfg)
    if any(part.get("fingerprint") != current or part.get("scope") != scope for part in parts):
        raise RuntimeError("Shard metadata mismatch")
    etas, spatial = parts[0]["etas"], parts[0]["spatial"]
    foreground_sum = sum((p["foreground_sum"] for p in parts), torch.zeros_like(parts[0]["foreground_sum"]))
    background_sum = sum((p["background_sum"] for p in parts), torch.zeros_like(parts[0]["background_sum"]))
    present_count = sum((p["present_count"] for p in parts), torch.zeros_like(parts[0]["present_count"]))
    image_count = sum(int(p["image_count"]) for p in parts)
    valid = present_count > 0
    foreground_mean = foreground_sum / present_count.clamp_min(1)[None, None]
    foreground_mean[..., ~valid] = float("nan")
    background_mean = background_sum / image_count
    miou = torch.cat((foreground_mean[..., valid], background_mean[..., None]), -1).mean(-1)
    best_flat = int(miou.argmax())
    result = {
        "etas": etas, "spatial": spatial,
        "foreground_mean": foreground_mean, "background_mean": background_mean, "miou": miou,
        "best_spatial": best_flat // etas.numel(), "best_eta": best_flat % etas.numel(),
        "image_count": image_count, "present_count": present_count,
        "candidate_chunk": min(int(p["candidate_chunk"]) for p in parts),
        "image_batch": [int(p["image_batch"]) for p in parts],
        "max_gpu_GiB": max(float(p["max_gpu_GiB"]) for p in parts),
    }
    return write_outputs(cfg, scope, dataset, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.merge_shards:
        print(merge_shards(cfg, args.scope, args.dataset, args.num_shards))
    elif args.shard_index is not None:
        if not args.device:
            parser.error("--device is required")
        print(run_shard(
            cfg, args.scope, args.dataset, args.device,
            args.shard_index, args.num_shards, force=args.force,
        ))
    else:
        if not args.device:
            parser.error("--device is required")
        print(run(cfg, args.scope, args.dataset, args.device, force=args.force))


if __name__ == "__main__":
    main()
