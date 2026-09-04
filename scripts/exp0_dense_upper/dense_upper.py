#!/usr/bin/env python3
"""Exact 0.1-grid dataset-level upper bound for EXP0.

This lives outside scripts/exp0 so adding the diagnostic does not invalidate
the already-completed EXP0 feature caches and policy checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ACTION_NAMES,
    fingerprint,
    fixed_action,
    load_config,
    output_root,
    selected_image_rows,
)
from ngsc_grpo.registry import get_spec


def decimal_values(stop_tenths: int) -> torch.Tensor:
    return torch.arange(stop_tenths + 1, dtype=torch.float32) / 10.0


def dense_axes() -> tuple[torch.Tensor, torch.Tensor]:
    etas = decimal_values(30)
    spatial = torch.tensor(
        list(itertools.product(decimal_values(10), decimal_values(10), decimal_values(40))),
        dtype=torch.float32,
    )
    if etas.numel() * spatial.shape[0] != 153_791:
        raise AssertionError((etas.shape, spatial.shape))
    return etas, spatial


def dense_root(cfg: Mapping, dataset: str) -> Path:
    return output_root(cfg) / "upper_bound_dense_0p1" / dataset


def _atomic_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.inference_mode()
def _exact_chunk(
    hat: torch.Tensor,
    affinity: torch.Tensor,
    distance: torch.Tensor,
    gt: torch.Tensor,
    grid_shape: tuple[int, int],
    spatial: torch.Tensor,
    etas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image IoU sums for K spatial actions and every eta.

    foreground: [K,E,C] summed over present images in this image batch.
    background: [K,E] summed over all images in this image batch.
    """
    batch, classes, _ = hat.shape
    actions = spatial.to(hat.device)
    tau = actions[:, 0]
    gamma = actions[:, 1]
    kappa = actions[:, 2]
    adjusted = affinity[:, None] * torch.exp(-kappa[None, :, None, None] * distance[:, None])
    low = adjusted < tau[None, :, None, None]
    refined = hat[:, None] * (1.0 - gamma[None, :, None, None] * low.to(hat.dtype))
    height, width = gt.shape[-2:]
    scores = F.interpolate(
        refined.reshape(batch * actions.shape[0] * classes, 1, *grid_shape),
        size=(height, width), mode="bilinear", align_corners=False,
    ).reshape(batch, actions.shape[0], classes, height * width)
    maximum, winner = scores.max(dim=2)
    gt_flat = gt.reshape(batch, classes, -1)
    gt_count = gt_flat.sum(-1)
    present = gt_count > 0
    gt_any = gt_flat.any(1)
    eta_count = int(etas.numel())
    bins = eta_count + 1
    # q is the number of eta thresholds satisfied by each maximum score.
    # Histogramming q and reverse-cumulating is exactly equivalent to forming
    # 31 full-resolution binary masks, without repeating the pixel work 31x.
    q = torch.bucketize(maximum.contiguous(), etas.to(hat.device), right=True)
    batch_candidates = batch * actions.shape[0]
    pixels = maximum.shape[-1]
    bk = torch.arange(batch_candidates, device=hat.device).reshape(
        batch, actions.shape[0], 1
    ).expand(-1, -1, pixels)
    class_group = bk * classes + winner
    class_hist_index = (class_group * bins + q).reshape(-1)
    prediction_hist = torch.zeros(
        batch_candidates * classes * bins, device=hat.device, dtype=torch.float32
    )
    prediction_hist.scatter_add_(
        0, class_hist_index, torch.ones_like(class_hist_index, dtype=torch.float32)
    )
    prediction_hist = prediction_hist.reshape(batch, actions.shape[0], classes, bins)

    winner_gt = torch.gather(
        gt_flat[:, None].expand(-1, actions.shape[0], -1, -1),
        2, winner[:, :, None],
    ).squeeze(2)
    intersection_hist = torch.zeros_like(prediction_hist).reshape(-1)
    intersection_hist.scatter_add_(0, class_hist_index, winner_gt.reshape(-1).float())
    intersection_hist = intersection_hist.reshape_as(prediction_hist)

    prediction_count = prediction_hist[..., 1:].flip(-1).cumsum(-1).flip(-1).double()
    intersection = intersection_hist[..., 1:].flip(-1).cumsum(-1).flip(-1).double()
    union = prediction_count + gt_count[:, None, :, None].double() - intersection
    iou = intersection / union.clamp_min(1.0)
    foreground = (iou * present[:, None, :, None]).sum(0).permute(0, 2, 1)

    any_hist_index = (bk * bins + q).reshape(-1)
    active_gt_hist = torch.zeros(
        batch_candidates * bins, device=hat.device, dtype=torch.float32
    )
    active_gt_hist.scatter_add_(
        0, any_hist_index,
        gt_any[:, None].expand(-1, actions.shape[0], -1).reshape(-1).float(),
    )
    active_gt_hist = active_gt_hist.reshape(batch, actions.shape[0], bins)
    active_gt_count = active_gt_hist[..., 1:].flip(-1).cumsum(-1).flip(-1).double()
    active_count = prediction_count.sum(2)
    gt_any_count = gt_any.sum(-1)[:, None, None].double()
    bg_intersection = pixels - active_count - gt_any_count + active_gt_count
    bg_union = pixels - active_gt_count
    background = (bg_intersection / bg_union.clamp_min(1.0)).sum(0)
    return foreground, background


def evaluate_exact_dense(
    cfg: Mapping,
    dataset: str,
    device: str,
    rows: Sequence[dict] | None = None,
    etas: torch.Tensor | None = None,
    spatial: torch.Tensor | None = None,
    candidate_chunk: int = 512,
    target_output_maps: int = 45_000,
) -> dict:
    run_device = torch.device(device)
    if rows is None:
        rows = selected_image_rows(cfg, dataset, 128, "internal")
    if etas is None or spatial is None:
        etas, spatial = dense_axes()
    etas = etas.float().cpu()
    spatial = spatial.float().cpu()
    classes = len(get_spec(dataset).foreground_classes)
    spatial_count, eta_count = spatial.shape[0], etas.numel()
    foreground_sum = torch.zeros(spatial_count, eta_count, classes, dtype=torch.float64, device=run_device)
    background_sum = torch.zeros(spatial_count, eta_count, dtype=torch.float64, device=run_device)
    present_count = torch.zeros(classes, dtype=torch.float64, device=run_device)
    image_count = 0
    active_chunk = min(int(candidate_chunk), spatial_count)
    image_batch = max(1, min(128, int(target_output_maps) // max(1, active_chunk * classes)))
    completed = 0
    torch.cuda.reset_peak_memory_stats(run_device)

    for image_start in range(0, len(rows), image_batch):
        image_rows = rows[image_start:image_start + image_batch]
        items = [torch.load(row["cache_path"], map_location="cpu") for row in image_rows]
        grid_shapes = {tuple(int(v) for v in item["meta"]["grid_shape"]) for item in items}
        if len(grid_shapes) != 1:
            raise ValueError(f"Mixed grid shapes: {grid_shapes}")
        grid_shape = next(iter(grid_shapes))
        hat = torch.stack([item["hat"].float() for item in items]).to(run_device)
        affinity = torch.stack([item["base_affinity"].float() for item in items]).to(run_device)
        coords = torch.stack([item["coords"].float() for item in items]).to(run_device)
        seeds = torch.stack([item["seed_idx"].long() for item in items]).to(run_device)
        gt = torch.stack([item["gt_masks"].bool() for item in items]).to(run_device)
        seed_coords = torch.gather(
            coords[:, None].expand(-1, classes, -1, -1), 2,
            seeds[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        distance = ((coords[:, None] - seed_coords[:, :, None]) ** 2).sum(-1)
        present_count += gt.flatten(2).any(-1).sum(0).double()

        action_start = 0
        while action_start < spatial_count:
            count = min(active_chunk, spatial_count - action_start)
            try:
                fg, bg = _exact_chunk(
                    hat, affinity, distance, gt, grid_shape,
                    spatial[action_start:action_start + count], etas,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if "out of memory" not in str(error).lower() or count <= 1:
                    raise
                active_chunk = max(1, count // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(json.dumps({
                    "dataset": dataset, "event": "dense_oom_retry",
                    "candidate_chunk": active_chunk, "image_batch": len(items),
                }), flush=True)
                continue
            foreground_sum[action_start:action_start + count] += fg
            background_sum[action_start:action_start + count] += bg
            action_start += count

        image_count += len(items)
        completed += len(items)
        if completed % 250 < len(items) or completed == len(rows):
            print(json.dumps({
                "dataset": dataset, "images": completed, "total": len(rows),
                "spatial_actions": spatial_count, "etas": eta_count,
                "candidate_chunk": active_chunk, "image_batch": image_batch,
                "max_gpu_GiB": round(torch.cuda.max_memory_allocated(run_device) / (1024 ** 3), 2),
            }), flush=True)

    valid_classes = present_count > 0
    foreground_mean = foreground_sum / present_count.clamp_min(1)[None, None]
    foreground_mean[..., ~valid_classes] = float("nan")
    background_mean = background_sum / max(1, image_count)
    miou = torch.cat(
        (foreground_mean[..., valid_classes], background_mean[..., None]), dim=-1
    ).mean(-1)
    best_flat = int(miou.argmax())
    best_spatial = best_flat // eta_count
    best_eta = best_flat % eta_count
    return {
        "etas": etas,
        "spatial": spatial,
        "foreground_mean": foreground_mean.cpu(),
        "background_mean": background_mean.cpu(),
        "foreground_sum": foreground_sum.cpu(),
        "background_sum": background_sum.cpu(),
        "miou": miou.cpu(),
        "best_spatial": best_spatial,
        "best_eta": best_eta,
        "image_count": image_count,
        "present_count": present_count.cpu(),
        "candidate_chunk": active_chunk,
        "image_batch": image_batch,
        "max_gpu_GiB": torch.cuda.max_memory_allocated(run_device) / (1024 ** 3),
    }


def _write_outputs(cfg: Mapping, dataset: str, result: Mapping) -> Path:
    root = dense_root(cfg, dataset)
    summary_path = root / "internal_summary.csv"
    metadata_path = root / "metadata.json"
    current = fingerprint(cfg)
    etas, spatial, miou = result["etas"], result["spatial"], result["miou"]
    best_spatial, best_eta = result["best_spatial"], result["best_eta"]
    tau, gamma, kappa = (float(v) for v in spatial[best_spatial])
    eta = float(etas[best_eta])
    class_names = list(get_spec(dataset).foreground_classes)
    foreground = result["foreground_mean"][best_spatial, best_eta]
    background = float(result["background_mean"][best_spatial, best_eta])
    best_miou = float(miou[best_spatial, best_eta])

    fixed = fixed_action(cfg, dataset).cpu()
    fixed_eta_idx = int(torch.isclose(etas, fixed[0], atol=1e-6).nonzero()[0])
    fixed_spatial_idx = int(torch.isclose(
        spatial, torch.tensor([fixed[1], fixed[2], fixed[3]])[None], atol=1e-6
    ).all(1).nonzero()[0])
    fixed_miou = float(miou[fixed_spatial_idx, fixed_eta_idx])

    grid_rows = []
    candidate_index = 0
    for spatial_idx, values in enumerate(spatial):
        tau_value, gamma_value, kappa_value = (float(v) for v in values)
        for eta_idx, eta_value in enumerate(etas):
            grid_rows.append({
                "candidate_index": candidate_index,
                "eta": float(eta_value),
                "tau": tau_value,
                "gamma": gamma_value,
                "kappa_sp": kappa_value,
                "exact_224_mIoU_percent": 100.0 * float(miou[spatial_idx, eta_idx]),
                "is_fixed_baseline": int(spatial_idx == fixed_spatial_idx and eta_idx == fixed_eta_idx),
                "is_selected": int(spatial_idx == best_spatial and eta_idx == best_eta),
            })
            candidate_index += 1
    _atomic_csv(root / "dense_grid_results.csv", grid_rows, list(grid_rows[0]))

    summary = {
        "dataset": dataset,
        "num_images": result["image_count"],
        "foreground_mIoU_percent": 100.0 * float(foreground[torch.isfinite(foreground)].mean()),
        "background_IoU_percent": 100.0 * background,
        "mIoU_percent": 100.0 * best_miou,
        "fixed_mIoU_percent": 100.0 * fixed_miou,
        "dense_minus_fixed_percent_point": 100.0 * (best_miou - fixed_miou),
        "eta": eta,
        "tau": tau,
        "gamma": gamma,
        "kappa_sp": kappa,
        "train_dataset": dataset,
        "train_pairs": 128,
        "split": "internal",
        "setting": "dense_grid_upper_bound_0p1",
        "method": cfg["method"],
        "fingerprint": current,
    }
    for index, name in enumerate(class_names):
        summary[f"class_{index}_name"] = name
        summary[f"class_{index}_IoU_percent"] = 100.0 * float(foreground[index])
    _atomic_csv(summary_path, [summary], list(summary))
    root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "exp0_fingerprint": current,
        "dataset": dataset,
        "grid_step": 0.1,
        "eta_values": int(etas.numel()),
        "tau_values": 11,
        "gamma_values": 11,
        "kappa_values": 41,
        "total_candidates": int(etas.numel() * spatial.shape[0]),
        "evaluation_resolution": [224, 224],
        "selection_metric": "exact final multiclass mIoU at 224x224",
        "internal_split_train_pairs": 128,
        "selected_action": dict(zip(ACTION_NAMES, (eta, tau, gamma, kappa))),
        "selected_mIoU_percent": 100.0 * best_miou,
        "fixed_mIoU_percent": 100.0 * fixed_miou,
        "candidate_chunk_final": result["candidate_chunk"],
        "image_batch": result["image_batch"],
        "max_gpu_GiB": result["max_gpu_GiB"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def run(cfg: Mapping, dataset: str, device: str, force: bool = False) -> Path:
    root = dense_root(cfg, dataset)
    summary_path = root / "internal_summary.csv"
    metadata_path = root / "metadata.json"
    current = fingerprint(cfg)
    if summary_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("exp0_fingerprint") == current and metadata.get("grid_step") == 0.1:
            return summary_path
    rows = selected_image_rows(cfg, dataset, 128, "internal")
    result = evaluate_exact_dense(cfg, dataset, device, rows=rows)
    return _write_outputs(cfg, dataset, result)


def run_shard(
    cfg: Mapping, dataset: str, device: str, shard_index: int, num_shards: int, force: bool = False
) -> Path:
    root = dense_root(cfg, dataset) / "shards"
    path = root / f"part_{shard_index}_of_{num_shards}.pt"
    current = fingerprint(cfg)
    if path.is_file() and not force:
        saved = torch.load(path, map_location="cpu")
        if saved.get("fingerprint") == current:
            return path
    all_rows = selected_image_rows(cfg, dataset, 128, "internal")
    rows = all_rows[shard_index::num_shards]
    result = evaluate_exact_dense(cfg, dataset, device, rows=rows)
    payload = {
        "fingerprint": current,
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


def merge_shards(cfg: Mapping, dataset: str, num_shards: int) -> Path:
    root = dense_root(cfg, dataset) / "shards"
    parts = [
        torch.load(root / f"part_{index}_of_{num_shards}.pt", map_location="cpu")
        for index in range(num_shards)
    ]
    current = fingerprint(cfg)
    if any(part.get("fingerprint") != current for part in parts):
        raise RuntimeError("Shard fingerprint mismatch")
    if any(part["num_shards"] != num_shards for part in parts):
        raise RuntimeError("Shard count mismatch")
    etas, spatial = parts[0]["etas"], parts[0]["spatial"]
    if any(not torch.equal(part["etas"], etas) or not torch.equal(part["spatial"], spatial)
           for part in parts[1:]):
        raise RuntimeError("Shard action axes mismatch")
    foreground_sum = sum((part["foreground_sum"] for part in parts), torch.zeros_like(parts[0]["foreground_sum"]))
    background_sum = sum((part["background_sum"] for part in parts), torch.zeros_like(parts[0]["background_sum"]))
    present_count = sum((part["present_count"] for part in parts), torch.zeros_like(parts[0]["present_count"]))
    image_count = sum(int(part["image_count"]) for part in parts)
    valid_classes = present_count > 0
    foreground_mean = foreground_sum / present_count.clamp_min(1)[None, None]
    foreground_mean[..., ~valid_classes] = float("nan")
    background_mean = background_sum / max(1, image_count)
    miou = torch.cat(
        (foreground_mean[..., valid_classes], background_mean[..., None]), dim=-1
    ).mean(-1)
    best_flat = int(miou.argmax())
    result = {
        "etas": etas,
        "spatial": spatial,
        "foreground_mean": foreground_mean,
        "background_mean": background_mean,
        "miou": miou,
        "best_spatial": best_flat // etas.numel(),
        "best_eta": best_flat % etas.numel(),
        "image_count": image_count,
        "present_count": present_count,
        "candidate_chunk": min(int(part["candidate_chunk"]) for part in parts),
        "image_batch": [int(part["image_batch"]) for part in parts],
        "max_gpu_GiB": max(float(part["max_gpu_GiB"]) for part in parts),
    }
    return _write_outputs(cfg, dataset, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.merge_shards:
        print(merge_shards(cfg, args.dataset, args.num_shards))
    elif args.shard_index is not None:
        if args.device is None:
            parser.error("--device is required for shard evaluation")
        print(run_shard(
            cfg, args.dataset, args.device, args.shard_index, args.num_shards, force=args.force
        ))
    else:
        if args.device is None:
            parser.error("--device is required")
        print(run(cfg, args.dataset, args.device, force=args.force))


if __name__ == "__main__":
    main()
