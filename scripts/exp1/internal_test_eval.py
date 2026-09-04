#!/usr/bin/env python3
"""Evaluate exp1 policies on source images unused by train/val.

This diagnostic is deliberately isolated under
``<experiment_root>/internal_test`` so it cannot be mistaken for a target-zero-shot
result or alter any cache used by the main feasibility experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ngsc_grpo.config import config_fingerprint, experiment_root, load_config, project_path
from ngsc_grpo.core import (
    apply_continuous_ngsc,
    extract_state,
    hard_masks_from_actions,
    original_image_normalize,
    upsample_patch_maps,
)
from ngsc_grpo.evaluation import (
    ActionProvider,
    _assemble_multiclass,
    _binary_auroc,
    _dice,
    _nanmean,
    _safe_iou,
)
from ngsc_grpo.registry import discover_records, get_spec, load_class_masks
from ngsc_grpo.splits import read_manifest


METHODS = ("MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP")


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _unused_records(cfg) -> list[dict]:
    split_root = experiment_root(cfg) / "splits"
    used = {
        row["image_relpath"]
        for name in ("source_train_manifest.csv", "source_val_manifest.csv")
        for row in read_manifest(split_root / name)
    }
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    prompt_map_dir = project_path(cfg, cfg["paths"]["prompt_map_dir"])
    records = []
    for dataset in cfg["sources"]:
        current = discover_records(data_root, prompt_map_dir, dataset, include_labels=True)
        for row in current:
            if row["image_relpath"] not in used:
                row["role"] = "internal_test"
                records.append(row)
    paths = [row["image_relpath"] for row in records]
    if len(paths) != len(set(paths)) or set(paths) & used:
        raise AssertionError("Internal-test paths are duplicated or overlap source train/val")
    return records


def prepare(cfg) -> Path:
    records = _unused_records(cfg)
    counts = defaultdict(int)
    for row in records:
        counts[row["dataset"]] += 1
    root = experiment_root(cfg) / "internal_test"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_fingerprint": config_fingerprint(cfg),
        "definition": "all source-pool images excluding source_train and source_val manifests",
        "num_images": len(records),
        "dataset_counts": dict(counts),
        "source_train_val_overlap": 0,
        "records": records,
    }
    path = root / "manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, ensure_ascii=False))
    return path


def _cache_name(row: dict) -> str:
    import hashlib

    digest = hashlib.sha1(row["image_relpath"].encode("utf-8")).hexdigest()[:12]
    return f"{Path(row['image_id']).stem}_{digest}.pt"


def cache_method(cfg, method: str, device: str, force: bool) -> Path:
    if method not in METHODS:
        raise ValueError(method)
    records = _unused_records(cfg)
    root = experiment_root(cfg) / "internal_test" / method / "cache"
    root.mkdir(parents=True, exist_ok=True)
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    fingerprint = config_fingerprint(cfg)
    dtype = torch.float16 if cfg["runtime"]["cache_dtype"] == "float16" else torch.float32
    extractor = None
    index_rows = []
    for row in tqdm(records, desc=f"internal-cache:{method}"):
        path = root / row["dataset"] / _cache_name(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        valid = False
        if path.is_file() and not force:
            try:
                old = torch.load(path, map_location="cpu")
                valid = (
                    old["meta"]["method"] == method
                    and old["meta"]["config_fingerprint"] == fingerprint
                    and old["meta"]["image_relpath"] == row["image_relpath"]
                    and old["meta"]["role"] == "internal_test"
                )
            except Exception:
                valid = False
        if not valid:
            if extractor is None:
                from ngsc_grpo.model_adapter import DenseBiomedCLIP

                extractor = DenseBiomedCLIP(cfg, method, device=device)
            image = Image.open(data_root / row["image_relpath"]).convert("RGB")
            frozen = extractor.frozen_ngsc_quantities(image, get_spec(row["dataset"]))
            class_names = list(get_spec(row["dataset"]).foreground_classes)
            states = torch.stack(
                [
                    extract_state(
                        frozen["raw"][idx], frozen["hat"][idx], frozen["base_affinity"][idx]
                    )
                    for idx in range(len(class_names))
                ]
            )
            payload = {
                "meta": {
                    "method": method,
                    "role": "internal_test",
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "image_relpath": row["image_relpath"],
                    "patient_id": row["patient_id"],
                    "class_names": class_names,
                    "grid_shape": list(frozen["grid_shape"]),
                    "image_size": [image.height, image.width],
                    "config_fingerprint": fingerprint,
                },
                "raw": frozen["raw"].detach().cpu().to(dtype),
                "hat": frozen["hat"].detach().cpu().to(dtype),
                "seed_idx": frozen["seed_idx"].detach().cpu().long(),
                "base_affinity": frozen["base_affinity"].detach().cpu().to(dtype),
                "coords": frozen["coords"].detach().cpu().to(dtype),
                "state": states.detach().cpu().float(),
            }
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        index_rows.append(
            {
                "dataset": row["dataset"],
                "image_id": row["image_id"],
                "image_relpath": row["image_relpath"],
                "cache_path": str(path.resolve()),
            }
        )
    index_path = root / "index.csv"
    _write_csv(index_path, index_rows)
    print(json.dumps({"method": method, "cached_images": len(index_rows), "index": str(index_path)}))
    return index_path


def _read_index(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_run(cfg, method: str, baseline: str, seed: int | None, force: bool) -> Path:
    name = f"{baseline}_seed{seed}" if baseline == "conditional_grpo" else baseline
    root = experiment_root(cfg) / "internal_test" / method
    result_path = root / "results" / f"{name}.csv"
    if result_path.is_file() and not force:
        return result_path
    index = _read_index(root / "cache" / "index.csv")
    provider = ActionProvider(cfg, method, baseline, seed, torch.device("cpu"))
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    per_pair = []
    for row in tqdm(index, desc=f"internal-eval:{method}:{name}"):
        item = torch.load(row["cache_path"], map_location="cpu")
        meta = item["meta"]
        if meta["role"] != "internal_test":
            raise AssertionError(f"Non-internal cache encountered: {row['cache_path']}")
        spec = get_spec(meta["dataset"])
        masks = load_class_masks(data_root, meta, size=None)
        original_hat = original_image_normalize(item["raw"].float())
        gt_masks = []
        score_maps = []
        thresholds = []
        for idx, class_name in enumerate(meta["class_names"]):
            action, _, _, _ = provider(item["state"][idx].float(), spec.radiology)
            hat = original_hat[idx] if baseline == "original_ngsc" else item["hat"][idx].float()
            _, score = hard_masks_from_actions(
                hat,
                item["base_affinity"][idx].float(),
                item["coords"].float(),
                int(item["seed_idx"][idx]),
                action,
                meta["grid_shape"],
                meta["image_size"],
            )
            gt_masks.append(masks[class_name].astype(bool))
            score_maps.append(score.numpy())
            thresholds.append(float(action[0]))
        predicted = _assemble_multiclass(
            np.stack(score_maps), np.asarray(thresholds, dtype=np.float32)
        )
        pred_any = np.logical_or.reduce(predicted)
        gt_any = np.logical_or.reduce(gt_masks)
        background_iou = _safe_iou(~pred_any, ~gt_any)
        for idx, class_name in enumerate(meta["class_names"]):
            pred = predicted[idx]
            gt = gt_masks[idx]
            per_pair.append(
                {
                    "dataset": meta["dataset"],
                    "image_id": meta["image_id"],
                    "class_name": class_name,
                    "gt_present": int(gt.any()),
                    "iou": _safe_iou(pred, gt),
                    "dice": _dice(pred, gt),
                    "auroc": _binary_auroc(score_maps[idx], gt),
                    "absent_fp_area": float(pred.mean()) if not gt.any() else float("nan"),
                    "background_iou": background_iou,
                }
            )
    summaries = []
    for dataset in cfg["sources"]:
        rows = [row for row in per_pair if row["dataset"] == dataset]
        foreground_by_class = []
        for class_name in get_spec(dataset).foreground_classes:
            values = [
                row["iou"]
                for row in rows
                if row["class_name"] == class_name and np.isfinite(row["iou"])
            ]
            foreground_by_class.append(float(np.mean(values)) if values else float("nan"))
        foreground_miou = _nanmean(foreground_by_class)
        backgrounds = {row["image_id"]: row["background_iou"] for row in rows}
        background_iou = _nanmean(backgrounds.values())
        valid = [value for value in foreground_by_class if np.isfinite(value)]
        summaries.append(
            {
                "dataset": dataset,
                "foreground_mIoU": foreground_miou,
                "background_IoU": background_iou,
                "mIoU": (sum(valid) + background_iou) / (len(valid) + 1),
                "foreground_Dice": _nanmean(row["dice"] for row in rows),
                "AUROC": _nanmean(row["auroc"] for row in rows),
                "absent_FP_area": _nanmean(row["absent_fp_area"] for row in rows),
                "num_images": len(backgrounds),
                "num_present_pairs": int(sum(row["gt_present"] for row in rows)),
            }
        )
    summaries.append(
        {
            "dataset": "macro_average",
            "foreground_mIoU": _nanmean(row["foreground_mIoU"] for row in summaries),
            "background_IoU": _nanmean(row["background_IoU"] for row in summaries),
            "mIoU": _nanmean(row["mIoU"] for row in summaries),
            "foreground_Dice": _nanmean(row["foreground_Dice"] for row in summaries),
            "AUROC": _nanmean(row["AUROC"] for row in summaries),
            "absent_FP_area": _nanmean(row["absent_FP_area"] for row in summaries),
            "num_images": int(sum(row["num_images"] for row in summaries)),
            "num_present_pairs": int(sum(row["num_present_pairs"] for row in summaries)),
        }
    )
    for row in summaries:
        for metric in (
            "foreground_mIoU",
            "background_IoU",
            "mIoU",
            "foreground_Dice",
            "AUROC",
            "absent_FP_area",
        ):
            value = float(row[metric])
            row[f"{metric}_percent"] = 100.0 * value if np.isfinite(value) else float("nan")
    _write_csv(result_path, summaries)
    _write_csv(root / "results" / f"{name}_per_pair.csv", per_pair)
    metadata = {
        "method": method,
        "run": name,
        "config_fingerprint": config_fingerprint(cfg),
        "split": "source pool minus source_train minus source_val",
        "num_images": len(index),
    }
    (root / "results" / f"{name}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result_path


def evaluate_method(cfg, method: str, force: bool) -> None:
    runs = [("original_ngsc", None), ("core_fixed", None), ("source_static", None)] + [
        ("conditional_grpo", seed) for seed in cfg["grpo"]["seeds"]
    ]
    root = experiment_root(cfg) / "internal_test" / method
    summary_path = root / "summary.csv"
    if summary_path.is_file() and not force:
        return
    index = _read_index(root / "cache" / "index.csv")
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    providers = {}
    per_run = {}
    for baseline, seed in runs:
        name = f"{baseline}_seed{seed}" if seed is not None else baseline
        providers[name] = (baseline, ActionProvider(cfg, method, baseline, seed, torch.device("cpu")))
        per_run[name] = []

    # Load each image and its masks once, then evaluate every baseline. This is
    # substantially faster than six independent passes over full-resolution masks.
    for row in tqdm(index, desc=f"internal-eval-all:{method}"):
        item = torch.load(row["cache_path"], map_location="cpu")
        meta = item["meta"]
        if meta["role"] != "internal_test":
            raise AssertionError(f"Non-internal cache encountered: {row['cache_path']}")
        spec = get_spec(meta["dataset"])
        mask_dict = load_class_masks(data_root, meta, size=None)
        gt_masks = [mask_dict[name].astype(bool) for name in meta["class_names"]]
        original_hat = original_image_normalize(item["raw"].float())
        score_maps_by_run = {name: [] for name in providers}
        thresholds_by_run = {name: [] for name in providers}
        provider_items = list(providers.items())
        for idx, _class_name in enumerate(meta["class_names"]):
            refined_maps = []
            for name, (baseline, provider) in provider_items:
                action, _, _, _ = provider(item["state"][idx].float(), spec.radiology)
                hat = original_hat[idx] if baseline == "original_ngsc" else item["hat"][idx].float()
                refined, _ = apply_continuous_ngsc(
                    hat,
                    item["base_affinity"][idx].float(),
                    item["coords"].float(),
                    int(item["seed_idx"][idx]),
                    action,
                )
                refined_maps.append(refined)
                thresholds_by_run[name].append(float(action[0]))
            # One batched interpolation for all six runs avoids repeating the
            # expensive full-resolution resize operation.
            upsampled = upsample_patch_maps(
                torch.stack(refined_maps), meta["grid_shape"], meta["image_size"]
            )
            for run_idx, (name, _) in enumerate(provider_items):
                score_maps_by_run[name].append(upsampled[run_idx].numpy())

        for name, _provider_spec in provider_items:
            score_maps = score_maps_by_run[name]
            thresholds = thresholds_by_run[name]
            predicted = _assemble_multiclass(
                np.stack(score_maps), np.asarray(thresholds, dtype=np.float32)
            )
            pred_any = np.logical_or.reduce(predicted)
            gt_any = np.logical_or.reduce(gt_masks)
            background_iou = _safe_iou(~pred_any, ~gt_any)
            for idx, class_name in enumerate(meta["class_names"]):
                pred = predicted[idx]
                gt = gt_masks[idx]
                per_run[name].append(
                    {
                        "dataset": meta["dataset"],
                        "image_id": meta["image_id"],
                        "class_name": class_name,
                        "gt_present": int(gt.any()),
                        "iou": _safe_iou(pred, gt),
                        "dice": _dice(pred, gt),
                        "absent_fp_area": float(pred.mean()) if not gt.any() else float("nan"),
                        "background_iou": background_iou,
                    }
                )

    all_summaries = []
    for name, pairs in per_run.items():
        summaries = []
        for dataset in cfg["sources"]:
            rows = [row for row in pairs if row["dataset"] == dataset]
            foreground_by_class = []
            for class_name in get_spec(dataset).foreground_classes:
                values = [
                    row["iou"]
                    for row in rows
                    if row["class_name"] == class_name and np.isfinite(row["iou"])
                ]
                foreground_by_class.append(float(np.mean(values)) if values else float("nan"))
            foreground_miou = _nanmean(foreground_by_class)
            backgrounds = {row["image_id"]: row["background_iou"] for row in rows}
            background_iou = _nanmean(backgrounds.values())
            valid = [value for value in foreground_by_class if np.isfinite(value)]
            summaries.append(
                {
                    "method": method,
                    "run": name,
                    "dataset": dataset,
                    "foreground_mIoU": foreground_miou,
                    "background_IoU": background_iou,
                    "mIoU": (sum(valid) + background_iou) / (len(valid) + 1),
                    "foreground_Dice": _nanmean(row["dice"] for row in rows),
                    "absent_FP_area": _nanmean(row["absent_fp_area"] for row in rows),
                    "num_images": len(backgrounds),
                    "num_present_pairs": int(sum(row["gt_present"] for row in rows)),
                }
            )
        summaries.append(
            {
                "method": method,
                "run": name,
                "dataset": "macro_average",
                "foreground_mIoU": _nanmean(row["foreground_mIoU"] for row in summaries),
                "background_IoU": _nanmean(row["background_IoU"] for row in summaries),
                "mIoU": _nanmean(row["mIoU"] for row in summaries),
                "foreground_Dice": _nanmean(row["foreground_Dice"] for row in summaries),
                "absent_FP_area": _nanmean(row["absent_FP_area"] for row in summaries),
                "num_images": int(sum(row["num_images"] for row in summaries)),
                "num_present_pairs": int(sum(row["num_present_pairs"] for row in summaries)),
            }
        )
        for summary in summaries:
            for metric in (
                "foreground_mIoU",
                "background_IoU",
                "mIoU",
                "foreground_Dice",
                "absent_FP_area",
            ):
                value = float(summary[metric])
                summary[f"{metric}_percent"] = 100.0 * value if np.isfinite(value) else float("nan")
        all_summaries.extend(summaries)
        _write_csv(root / "results" / f"{name}_per_pair.csv", pairs)
    _write_csv(summary_path, all_summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feasibility.yaml")
    parser.add_argument("command", choices=("prepare", "cache", "evaluate"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare(cfg)
        return
    if args.method is None:
        parser.error(f"{args.command} requires --method")
    if args.command == "cache":
        cache_method(cfg, args.method, args.device, args.force)
    else:
        evaluate_method(cfg, args.method, args.force)


if __name__ == "__main__":
    main()
