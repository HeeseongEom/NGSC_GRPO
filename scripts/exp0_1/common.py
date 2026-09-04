#!/usr/bin/env python3
"""Shared EXP0 split, cache, policy, reward, training, and evaluation utilities."""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from PIL import Image
from torch import nn
from torch.distributions import Beta, kl_divergence

from ngsc_grpo.core import normalized_coords, per_class_normalize, upsample_patch_maps
from ngsc_grpo.evaluation import _assemble_multiclass, _dice, _nanmean, _safe_iou
from ngsc_grpo.registry import discover_records, get_spec, load_class_masks


ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = ("eta", "tau", "gamma", "kappa_sp")


def load_config(path: str | Path = ROOT / "configs" / "exp0_1.yaml") -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping) -> None:
    datasets = tuple(cfg["datasets"])
    if len(datasets) != 8 or len(set(datasets)) != 8:
        raise ValueError("EXP0 requires exactly eight unique datasets")
    for name in datasets:
        get_spec(name)
    if tuple(cfg["actions"]["names"]) != ACTION_NAMES:
        raise ValueError(f"EXP0 full action must be {ACTION_NAMES}")
    if sorted(int(v) for v in cfg["split"]["train_pair_counts"]) != [32, 128]:
        raise ValueError("train_pair_counts must be [32, 128]")
    if sorted(int(v) for v in cfg["optimization"]["group_sizes"]) != [4, 8, 16]:
        raise ValueError("group_sizes must be [4, 8, 16]")
    for name in ACTION_NAMES:
        low, high = cfg["actions"][name]
        if not float(low) < float(high):
            raise ValueError(f"Invalid action bounds for {name}")
    weights = [
        float(cfg["reward"]["dice_weight"]),
        float(cfg["reward"]["lesion_f1_weight"]),
        float(cfg["reward"]["boundary_f1_weight"]),
    ]
    if any(value < 0 for value in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Reward weights must be nonnegative and sum to one: {weights}")
    margin = float(cfg["reward"]["absent_margin"])
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"absent_margin must be in [0,1): {margin}")


def output_root(cfg: Mapping) -> Path:
    return ROOT / cfg["experiment"]["output_root"] / cfg["experiment"]["name"]


def artifact_root(cfg: Mapping) -> Path:
    return ROOT / cfg["experiment"]["output_root"] / cfg["artifacts"]["reuse_experiment"]


def data_root(cfg: Mapping) -> Path:
    return ROOT / cfg["paths"]["data_root"]


def prompt_map_root(cfg: Mapping) -> Path:
    return ROOT / cfg["paths"]["prompt_map_dir"]


def fingerprint(cfg: Mapping) -> str:
    payload = {key: value for key, value in cfg.items() if not key.startswith("_")}
    # Throughput controls do not change features, rewards, policies, or metrics.
    # Excluding them lets a safe OOM retry/tuning change reuse identical artifacts.
    payload = json.loads(json.dumps(payload))
    for key in ("batch_size", "min_batch_size"):
        payload.get("cache", {}).pop(key, None)
        payload.get("evaluation", {}).pop(key, None)
    payload.pop("runtime", None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    for path in (
        ROOT / "src" / "ngsc_grpo" / "core.py",
        ROOT / "src" / "ngsc_grpo" / "model_adapter.py",
        ROOT / "src" / "ngsc_grpo" / "registry.py",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def records_for_dataset(cfg: Mapping, dataset: str) -> list[dict]:
    return discover_records(data_root(cfg), prompt_map_root(cfg), dataset, include_labels=True)


def split_dir(cfg: Mapping, dataset: str, train_pairs: int) -> Path:
    return artifact_root(cfg) / "splits" / f"n{int(train_pairs)}" / dataset


def _quota(total: int, names: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(int(total), len(names))
    return {name: base + int(index < remainder) for index, name in enumerate(names)}


def _balanced_prompt_sample(
    records: Sequence[dict], class_name: str, count: int, rng: random.Random, balance: bool
) -> list[dict]:
    positive = [row for row in records if class_name in row["present_classes"]]
    absent = [row for row in records if class_name not in row["present_classes"]]
    rng.shuffle(positive)
    rng.shuffle(absent)
    if balance and positive and absent:
        positive_count = min(len(positive), (count + 1) // 2)
        absent_count = min(len(absent), count - positive_count)
        selected = positive[:positive_count] + absent[:absent_count]
        used = {(row["image_id"], class_name) for row in selected}
        remainder = [row for row in positive[positive_count:] + absent[absent_count:]
                     if (row["image_id"], class_name) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    else:
        pool = list(records)
        rng.shuffle(pool)
        selected = pool[:count]
    if len(selected) != count:
        raise RuntimeError(
            f"Cannot sample {count} unique pairs for prompt {class_name!r}; available={len(records)}"
        )
    rng.shuffle(selected)
    return selected


def build_dataset_split(cfg: Mapping, dataset: str, train_pair_count: int, force: bool = False) -> Path:
    root = split_dir(cfg, dataset, train_pair_count)
    pair_path = root / "train_pairs.csv"
    internal_path = root / "internal_images.csv"
    meta_path = root / "metadata.json"
    current = fingerprint(cfg)
    if cfg.get("artifacts", {}).get("reuse_experiment"):
        required = (pair_path, internal_path, meta_path)
        if force:
            raise RuntimeError("EXP0_1 reuses frozen EXP0 splits; --force is not allowed")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"Missing reused EXP0 split artifacts under {root}")
        return root
    if not force and pair_path.is_file() and internal_path.is_file() and meta_path.is_file():
        if json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") == current:
            return root

    records = records_for_dataset(cfg, dataset)
    class_names = list(get_spec(dataset).foreground_classes)
    quotas = _quota(train_pair_count, class_names)
    seed = int(cfg["experiment"]["seed"]) + 1009 * list(cfg["datasets"]).index(dataset) + train_pair_count
    pairs = []
    for class_index, class_name in enumerate(class_names):
        rng = random.Random(seed + 97 * class_index)
        selected = _balanced_prompt_sample(
            records,
            class_name,
            quotas[class_name],
            rng,
            bool(cfg["split"]["balance_presence_when_possible"]),
        )
        for row in selected:
            pairs.append({
                "dataset": dataset,
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_relpath": row["image_relpath"],
                "class_name": class_name,
                "present": int(class_name in row["present_classes"]),
            })
    random.Random(seed + 7919).shuffle(pairs)
    if len(pairs) != train_pair_count:
        raise AssertionError((dataset, len(pairs), train_pair_count))

    selected_units = {row["patient_id"] for row in pairs}
    internal = [row for row in records if row["patient_id"] not in selected_units]
    if not internal:
        raise RuntimeError(f"Split leaves no internal images for {dataset}, n={train_pair_count}")
    internal_rows = [{
        "dataset": dataset,
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "image_relpath": row["image_relpath"],
    } for row in internal]
    write_csv(pair_path, pairs)
    write_csv(internal_path, internal_rows)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "fingerprint": current,
        "dataset": dataset,
        "seed": seed,
        "requested_train_pairs": train_pair_count,
        "actual_train_pairs": len(pairs),
        "train_unique_images": len({row["image_id"] for row in pairs}),
        "train_unique_patient_or_image_units": len(selected_units),
        "internal_images": len(internal_rows),
        "prompt_quotas": quotas,
        "prompt_positive_counts": {
            name: sum(int(row["present"]) for row in pairs if row["class_name"] == name)
            for name in class_names
        },
        "image_leakage": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def cache_dir(cfg: Mapping, dataset: str) -> Path:
    return artifact_root(cfg) / "cache" / cfg["method"] / dataset


def cache_index(cfg: Mapping, dataset: str) -> Path:
    return cache_dir(cfg, dataset) / "index.csv"


def _cache_file_name(image_relpath: str) -> str:
    digest = hashlib.sha1(image_relpath.encode("utf-8")).hexdigest()[:12]
    return f"{Path(image_relpath).stem}_{digest}.pt"


def build_dataset_cache(cfg: Mapping, dataset: str, device: str, force: bool = False) -> Path:
    from ngsc_grpo.model_adapter import DenseBiomedCLIP

    records = records_for_dataset(cfg, dataset)
    root = cache_dir(cfg, dataset)
    if cfg.get("artifacts", {}).get("reuse_experiment"):
        index = root / "index.csv"
        if force:
            raise RuntimeError("EXP0_1 reuses frozen EXP0 caches; --force is not allowed")
        if not index.is_file():
            raise FileNotFoundError(f"Missing reused EXP0 cache index: {index}")
        return index
    root.mkdir(parents=True, exist_ok=True)
    current = fingerprint(cfg)
    extractor = DenseBiomedCLIP({
        **cfg,
        "dense_methods": ["MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP"],
        "runtime": {"device": device},
        "_project_root": str(ROOT),
    }, cfg["method"], device=device)
    spec = get_spec(dataset)
    class_text, cdam_text = extractor.text_features(spec)
    text_delta = F.normalize(class_text[:-1] - class_text[-1].unsqueeze(0), dim=-1).cpu().float()
    torch.save({"class_names": list(spec.foreground_classes), "text_delta": text_delta}, root / "text_delta.pt")
    dtype = torch.float16 if cfg["cache"]["dtype"] == "float16" else torch.float32
    reward_size = int(cfg["cache"]["reward_resolution"])
    index_rows = []
    pending = []
    for row in records:
        path = root / _cache_file_name(row["image_relpath"])
        valid = False
        if path.is_file() and not force:
            try:
                saved = torch.load(path, map_location="cpu")
                valid = saved["meta"].get("fingerprint") == current
            except Exception:
                valid = False
        if not valid:
            pending.append((row, path))
        index_rows.append({
            "dataset": dataset,
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_relpath": row["image_relpath"],
            "cache_path": str(path.resolve()),
        })

    requested_batch = int(cfg["cache"].get("batch_size", cfg["runtime"].get("cache_batch_size", 1)))
    minimum_batch = int(cfg["cache"].get("min_batch_size", 1))
    batch_size = max(minimum_batch, requested_batch)
    completed = len(records) - len(pending)
    start = 0
    while start < len(pending):
        count = min(batch_size, len(pending) - start)
        chunk = pending[start:start + count]
        images = []
        original_sizes = []
        try:
            for row, _ in chunk:
                with Image.open(data_root(cfg) / row["image_relpath"]) as source:
                    image = source.convert("RGB")
                original_sizes.append((image.height, image.width))
                images.append(image)
            with torch.inference_mode():
                cls_batch, local_batch, grid_shape = extractor.image_features_batch(images)
                scores_batch = local_batch @ class_text.T
                raw_batch = scores_batch[..., :-1].transpose(1, 2) - scores_batch[..., -1].unsqueeze(1)
                flat = raw_batch.flatten(0, 1)
                hat_batch = per_class_normalize(flat).reshape_as(raw_batch)
                seeds_batch = hat_batch.argmax(dim=-1)
                coords = normalized_coords(
                    *grid_shape, device=local_batch.device, dtype=local_batch.dtype
                )

                for offset, (row, path) in enumerate(chunk):
                    local = local_batch[offset]
                    cdam_features = torch.cat((cls_batch[offset:offset + 1], local), dim=0)
                    cdam = extractor._cdam_matrix(
                        cdam_features,
                        cdam_text,
                        cfg["ngsc_core"]["cdam_temperature"],
                        cfg["ngsc_core"]["cdam_softmax_temperature"],
                    )
                    seeds = seeds_batch[offset]
                    affinity = torch.stack([cdam[int(seed) + 1] for seed in seeds])
                    masks = load_class_masks(data_root(cfg), row, size=(reward_size, reward_size))
                    gt = torch.from_numpy(
                        np.stack([masks[name] for name in spec.foreground_classes])
                    ).bool()
                    payload = {
                        "meta": {
                            "fingerprint": current,
                            "method": cfg["method"],
                            "dataset": dataset,
                            "image_id": row["image_id"],
                            "patient_id": row["patient_id"],
                            "image_relpath": row["image_relpath"],
                            "class_names": list(spec.foreground_classes),
                            "grid_shape": list(grid_shape),
                            "eval_size": [reward_size, reward_size],
                            "original_image_size": list(original_sizes[offset]),
                        },
                        "local": local.detach().cpu().to(dtype),
                        "hat": hat_batch[offset].detach().cpu().to(dtype),
                        "seed_idx": seeds.detach().cpu().long(),
                        "base_affinity": affinity.detach().cpu().to(dtype),
                        "coords": coords.detach().cpu().to(dtype),
                        "gt_masks": gt,
                        "present": gt.flatten(1).any(1),
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(path)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower() or count <= minimum_batch:
                raise
            batch_size = max(minimum_batch, count // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({
                "dataset": dataset, "event": "cache_oom_retry", "batch_size": batch_size,
            }), flush=True)
            continue
        finally:
            for image in images:
                image.close()
        start += count
        completed += count
        if completed % 100 < count or completed == len(records):
            memory = torch.cuda.max_memory_allocated(extractor.device) / (1024 ** 3)
            print(json.dumps({
                "dataset": dataset, "cached": completed, "total": len(records),
                "batch_size": count, "max_gpu_GiB": round(memory, 2),
            }), flush=True)
    write_csv(cache_index(cfg, dataset), index_rows)
    (root / "metadata.json").write_text(json.dumps({
        "fingerprint": current,
        "dataset": dataset,
        "method": cfg["method"],
        "num_images": len(index_rows),
        "representation": "pre-SCM dense local image embedding",
    }, indent=2), encoding="utf-8")
    return cache_index(cfg, dataset)


def index_lookup(cfg: Mapping, dataset: str) -> dict[str, dict]:
    path = cache_index(cfg, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache for {dataset}: {path}")
    return {row["image_id"]: row for row in read_csv(path)}


def load_text_delta(cfg: Mapping, dataset: str) -> torch.Tensor:
    payload = torch.load(cache_dir(cfg, dataset) / "text_delta.pt", map_location="cpu")
    expected = list(get_spec(dataset).foreground_classes)
    if payload["class_names"] != expected:
        raise RuntimeError(f"Text delta class order mismatch for {dataset}")
    return payload["text_delta"].float()


@dataclass
class Pair:
    dataset: str
    image_id: str
    class_name: str
    class_idx: int
    present: bool
    local: torch.Tensor
    text_delta: torch.Tensor
    hat: torch.Tensor
    seed_idx: int
    base_affinity: torch.Tensor
    coords: torch.Tensor
    gt_mask: torch.Tensor
    grid_shape: tuple[int, int]
    gt_component_labels: np.ndarray | None = None
    gt_component_count: int = 0


def load_train_pairs(cfg: Mapping, dataset: str, train_pair_count: int) -> list[Pair]:
    split_path = split_dir(cfg, dataset, train_pair_count) / "train_pairs.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split: {split_path}")
    lookup = index_lookup(cfg, dataset)
    text_delta = load_text_delta(cfg, dataset)
    class_names = list(get_spec(dataset).foreground_classes)
    payloads = {}
    pairs = []
    for row in read_csv(split_path):
        image_id = row["image_id"]
        if image_id not in payloads:
            payloads[image_id] = torch.load(lookup[image_id]["cache_path"], map_location="cpu")
        item = payloads[image_id]
        class_idx = class_names.index(row["class_name"])
        cached_present = bool(item["present"][class_idx])
        manifest_present = bool(int(row["present"]))
        if cached_present != manifest_present:
            raise RuntimeError(
                f"Split/cache presence mismatch: {dataset}/{image_id}/{row['class_name']}"
            )
        gt_mask = item["gt_masks"][class_idx].bool()
        component_count, component_labels = cv2.connectedComponents(
            gt_mask.numpy().astype(np.uint8, copy=False), connectivity=8
        )
        pairs.append(Pair(
            dataset=dataset,
            image_id=image_id,
            class_name=row["class_name"],
            class_idx=class_idx,
            present=cached_present,
            local=item["local"].float(),
            text_delta=text_delta[class_idx],
            hat=item["hat"][class_idx].float(),
            seed_idx=int(item["seed_idx"][class_idx]),
            base_affinity=item["base_affinity"][class_idx].float(),
            coords=item["coords"].float(),
            gt_mask=gt_mask,
            grid_shape=tuple(int(v) for v in item["meta"]["grid_shape"]),
            gt_component_labels=component_labels,
            gt_component_count=int(component_count - 1),
        ))
    if len(pairs) != train_pair_count:
        raise AssertionError((dataset, len(pairs), train_pair_count))
    return pairs


def action_bounds(cfg: Mapping, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor([cfg["actions"][name][0] for name in ACTION_NAMES], device=device, dtype=dtype)
    high = torch.tensor([cfg["actions"][name][1] for name in ACTION_NAMES], device=device, dtype=dtype)
    return low, high


def map_actions(normalized: torch.Tensor, cfg: Mapping) -> torch.Tensor:
    low, high = action_bounds(cfg, normalized.device, normalized.dtype)
    return low + (high - low) * normalized


def fixed_action(cfg: Mapping, dataset: str, device=None) -> torch.Tensor:
    spec = get_spec(dataset)
    core = cfg["ngsc_core"]
    return torch.tensor([
        core["radiology_eta"] if spec.radiology else core["nonradiology_eta"],
        core["original_tau"], core["original_gamma"], 0.0,
    ], dtype=torch.float32, device=device)


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def reference_beta(cfg: Mapping, dataset: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = action_bounds(cfg, device)
    mean = ((fixed_action(cfg, dataset, device) - low) / (high - low)).clamp(0.02, 0.98)
    concentration = float(cfg["policy"]["reference_concentration"])
    return mean * concentration, (1.0 - mean) * concentration


class GlobalBetaPolicy(nn.Module):
    def __init__(self, action_dim: int, beta_floor: float, beta_max: float):
        super().__init__()
        self.beta_floor = float(beta_floor)
        self.beta_max = float(beta_max)
        self.raw = nn.Parameter(torch.zeros(action_dim, 2))

    def initialize(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        target = torch.stack((alpha, beta), dim=-1) - self.beta_floor
        self.raw.data.copy_(inverse_softplus(target.clamp_min(1e-6)))

    def parameters_ab(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = (self.beta_floor + F.softplus(self.raw)).clamp(max=self.beta_max)
        values = values.unsqueeze(0).expand(int(batch), -1, -1)
        return values[..., 0], values[..., 1]


class CNNBetaPolicy(nn.Module):
    """Small controller over prompt-conditioned dense image representations before SCM."""

    def __init__(self, embedding_dim: int, hidden: int, action_dim: int, beta_floor: float, beta_max: float):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.beta_floor = float(beta_floor)
        self.beta_max = float(beta_max)
        self.reduce = nn.Conv2d(self.embedding_dim, self.hidden, kernel_size=1)
        self.spatial = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1, groups=self.hidden)
        self.mix = nn.Conv2d(self.hidden, self.hidden, kernel_size=1)
        self.head = nn.Linear(self.hidden * 2, self.action_dim * 2)

    def initialize(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        target = torch.stack((alpha, beta), dim=-1) - self.beta_floor
        nn.init.zeros_(self.head.weight)
        self.head.bias.data.copy_(inverse_softplus(target.clamp_min(1e-6)).reshape(-1))

    def parameters_ab(
        self, local: torch.Tensor, text_delta: torch.Tensor, grid_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, dimension = local.shape
        height, width = grid_shape
        if dimension != self.embedding_dim or patches != height * width:
            raise ValueError((local.shape, self.embedding_dim, grid_shape))
        conditioned = local * text_delta[:, None, :]
        conditioned = conditioned.transpose(1, 2).reshape(batch, dimension, height, width)
        feature = F.gelu(self.reduce(conditioned))
        feature = F.gelu(self.mix(F.gelu(self.spatial(feature))) + feature)
        pooled = torch.cat((feature.mean((-2, -1)), feature.amax((-2, -1))), dim=-1)
        raw = self.head(pooled).reshape(batch, self.action_dim, 2)
        values = (self.beta_floor + F.softplus(raw)).clamp(max=self.beta_max)
        return values[..., 0], values[..., 1]


def refined_patch_scores(pair: Pair, actions: torch.Tensor) -> torch.Tensor:
    eta, tau, gamma, kappa = actions.unbind(-1)
    del eta
    coords = pair.coords.to(actions.device)
    distance = ((coords - coords[pair.seed_idx]) ** 2).sum(-1)
    affinity = pair.base_affinity.to(actions.device)[None] * torch.exp(-kappa[:, None] * distance[None])
    suppressed = affinity < tau[:, None]
    return pair.hat.to(actions.device)[None] * (1.0 - gamma[:, None] * suppressed.to(actions.dtype))


def _boundary_f1(prediction: torch.Tensor, gt: torch.Tensor, tolerance: int) -> torch.Tensor:
    """Boundary F1 for a [G,H,W] prediction batch and one [H,W] target."""
    pred = prediction.bool()
    target = gt.bool().expand_as(pred)

    def boundary(mask: torch.Tensor) -> torch.Tensor:
        values = mask.float().unsqueeze(1)
        padded = F.pad(values, (1, 1, 1, 1), value=0.0)
        eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
        return mask & ~eroded[:, 0].bool()

    pred_boundary = boundary(pred)
    gt_boundary = boundary(target)
    kernel = 2 * int(tolerance) + 1
    pred_neighborhood = F.max_pool2d(
        pred_boundary.float().unsqueeze(1), kernel, stride=1, padding=int(tolerance)
    )[:, 0].bool()
    gt_neighborhood = F.max_pool2d(
        gt_boundary.float().unsqueeze(1), kernel, stride=1, padding=int(tolerance)
    )[:, 0].bool()
    pred_count = pred_boundary.sum((-2, -1)).float()
    gt_count = gt_boundary.sum((-2, -1)).float()
    precision = (pred_boundary & gt_neighborhood).sum((-2, -1)).float() / pred_count.clamp_min(1.0)
    recall = (gt_boundary & pred_neighborhood).sum((-2, -1)).float() / gt_count.clamp_min(1.0)
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)


def _lesion_f1(
    pair: Pair,
    prediction: torch.Tensor,
    connectivity: int,
    minimum_overlap: int,
) -> torch.Tensor:
    """One-to-one component F1; components match when overlap reaches the configured pixels."""
    gt_labels = pair.gt_component_labels
    gt_count = int(pair.gt_component_count)
    if gt_labels is None:
        labels_total, gt_labels = cv2.connectedComponents(
            pair.gt_mask.numpy().astype(np.uint8, copy=False), connectivity=int(connectivity)
        )
        gt_count = int(labels_total - 1)
    predictions = prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
    scores = []
    for mask in predictions:
        labels_total, pred_labels = cv2.connectedComponents(mask, connectivity=int(connectivity))
        pred_count = int(labels_total - 1)
        if pred_count == 0 or gt_count == 0:
            scores.append(0.0)
            continue
        joint = np.bincount(
            gt_labels.reshape(-1).astype(np.int64) * labels_total
            + pred_labels.reshape(-1).astype(np.int64),
            minlength=(gt_count + 1) * labels_total,
        ).reshape(gt_count + 1, labels_total)[1:, 1:]
        candidates = np.argwhere(joint >= int(minimum_overlap))
        if candidates.size == 0:
            true_positive = 0
        else:
            order = sorted(candidates.tolist(), key=lambda ij: int(joint[ij[0], ij[1]]), reverse=True)
            used_gt, used_pred = set(), set()
            true_positive = 0
            for gt_index, pred_index in order:
                if gt_index in used_gt or pred_index in used_pred:
                    continue
                used_gt.add(gt_index)
                used_pred.add(pred_index)
                true_positive += 1
        scores.append(2.0 * true_positive / (gt_count + pred_count))
    return torch.tensor(scores, dtype=torch.float32, device=prediction.device)


def reward_with_components(
    pair: Pair, actions: torch.Tensor, reward_cfg: Mapping
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    refined = refined_patch_scores(pair, actions)
    score = upsample_patch_maps(refined, pair.grid_shape, pair.gt_mask.shape[-2:])
    prediction = score >= actions[:, 0, None, None]
    empty = ~prediction.flatten(1).any(1)
    if pair.present:
        gt = pair.gt_mask.to(actions.device)[None]
        intersection = (prediction & gt).sum((-2, -1)).float()
        denominator = prediction.sum((-2, -1)).float() + gt.sum().float()
        dice = 2.0 * intersection / denominator.clamp_min(1e-8)
        lesion_f1 = _lesion_f1(
            pair, prediction,
            int(reward_cfg["lesion_connectivity"]),
            int(reward_cfg["lesion_min_overlap_pixels"]),
        )
        boundary_f1 = _boundary_f1(
            prediction, gt, int(reward_cfg["boundary_tolerance_pixels"])
        )
        reward = (
            float(reward_cfg["dice_weight"]) * dice
            + float(reward_cfg["lesion_f1_weight"]) * lesion_f1
            + float(reward_cfg["boundary_f1_weight"]) * boundary_f1
        )
        nan = torch.full_like(reward, float("nan"))
        return reward, {
            "dice": dice,
            "lesion_f1": lesion_f1,
            "boundary_f1": boundary_f1,
            "absent_reward": nan,
            "empty_prediction": empty.float(),
        }
    area = prediction.float().mean((-2, -1))
    margin = float(reward_cfg["absent_margin"])
    reward = torch.where(empty, torch.ones_like(area), (1.0 - area) * (1.0 - margin))
    nan = torch.full_like(reward, float("nan"))
    return reward, {
        "dice": nan,
        "lesion_f1": nan,
        "boundary_f1": nan,
        "absent_reward": reward,
        "empty_prediction": empty.float(),
    }


def hard_reward(
    pair: Pair, actions: torch.Tensor, reward_cfg: Mapping | None = None
) -> torch.Tensor:
    if reward_cfg is None:
        reward_cfg = {
            "dice_weight": 0.6,
            "lesion_f1_weight": 0.3,
            "boundary_f1_weight": 0.1,
            "lesion_connectivity": 8,
            "lesion_min_overlap_pixels": 1,
            "boundary_tolerance_pixels": 2,
            "absent_margin": 0.2,
        }
    return reward_with_components(pair, actions, reward_cfg)[0]


def policy_distribution(
    policy: nn.Module,
    kind: str,
    batch: Sequence[Pair],
    device: torch.device,
) -> Beta:
    if kind == "global":
        alpha, beta = policy.parameters_ab(len(batch))
    elif kind == "cnn":
        grid_shapes = {pair.grid_shape for pair in batch}
        if len(grid_shapes) != 1:
            raise ValueError(f"CNN batch has mixed grid shapes: {grid_shapes}")
        local = torch.stack([pair.local for pair in batch]).to(device)
        text = torch.stack([pair.text_delta for pair in batch]).to(device)
        alpha, beta = policy.parameters_ab(local, text, next(iter(grid_shapes)))
    else:
        raise ValueError(kind)
    return Beta(alpha, beta)


def run_dir(cfg: Mapping, dataset: str, train_pairs: int, group_size: int, kind: str) -> Path:
    return output_root(cfg) / "runs" / dataset / f"n{train_pairs}" / f"g{group_size}" / kind


def train_policy(
    cfg: Mapping,
    dataset: str,
    train_pair_count: int,
    group_size: int,
    kind: str,
    device: str,
    force: bool = False,
) -> Path:
    destination = run_dir(cfg, dataset, train_pair_count, group_size, kind)
    checkpoint = destination / "policy_final.pt"
    current_fingerprint = fingerprint(cfg)
    if checkpoint.is_file() and not force:
        saved = torch.load(checkpoint, map_location="cpu")
        if saved.get("fingerprint") == current_fingerprint:
            return checkpoint

    run_device = torch.device(device)
    seed = int(cfg["experiment"]["seed"]) + 1009 * list(cfg["datasets"]).index(dataset)
    seed += 13 * int(train_pair_count) + 31 * int(group_size) + (0 if kind == "global" else 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pairs = load_train_pairs(cfg, dataset, train_pair_count)
    settings = cfg["optimization"]
    alpha_ref, beta_ref = reference_beta(cfg, dataset, run_device)
    beta_floor = float(cfg["policy"]["beta_floor"])
    beta_max = float(cfg["policy"]["beta_max"])
    if kind == "global":
        policy = GlobalBetaPolicy(4, beta_floor, beta_max).to(run_device)
    elif kind == "cnn":
        policy = CNNBetaPolicy(
            pairs[0].local.shape[-1], int(cfg["policy"]["cnn_hidden_channels"]),
            4, beta_floor, beta_max,
        ).to(run_device)
    else:
        raise ValueError(kind)
    policy.initialize(alpha_ref, beta_ref)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["lr"]))
    rng = random.Random(seed + 100_003)
    logs = []

    for update in range(1, int(settings["updates"]) + 1):
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(int(settings["batch_pairs"]))]
        with torch.no_grad():
            old = policy_distribution(policy, kind, batch, run_device)
            frozen = Beta(old.concentration1.detach().clone(), old.concentration0.detach().clone())
            normalized = frozen.sample((int(group_size),)).permute(1, 0, 2).contiguous()
            old_log_prob = frozen.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
            physical = map_actions(normalized, cfg)
            reward_outputs = [
                reward_with_components(pair, physical[index], cfg["reward"])
                for index, pair in enumerate(batch)
            ]
            rewards = torch.stack([value[0] for value in reward_outputs])
            reward_components = {
                name: torch.stack([value[1][name] for value in reward_outputs])
                for name in ("dice", "lesion_f1", "boundary_f1", "absent_reward", "empty_prediction")
            }
            centered = rewards - rewards.mean(1, keepdim=True)
            spread = rewards.std(1, keepdim=True, unbiased=False)
            advantages = torch.where(
                spread > 1e-6, centered / (spread + 1e-6), torch.zeros_like(centered)
            )

        epoch_loss, epoch_kl, epoch_clip, epoch_ratio, epoch_grad = [], [], [], [], []
        for _ in range(int(settings["ppo_epochs"])):
            current = policy_distribution(policy, kind, batch, run_device)
            log_prob = current.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = ratio.clamp(
                1.0 - float(settings["clip_epsilon"]), 1.0 + float(settings["clip_epsilon"])
            )
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            reference = Beta(alpha_ref, beta_ref)
            reference_kl = kl_divergence(current, reference).sum(-1).mean()
            loss = policy_loss + float(settings["kl_beta"]) * reference_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip"]))
            optimizer.step()
            epoch_loss.append(float(policy_loss.detach()))
            epoch_kl.append(float(reference_kl.detach()))
            epoch_clip.append(float(((ratio < 1.0 - float(settings["clip_epsilon"])) |
                                     (ratio > 1.0 + float(settings["clip_epsilon"]))).float().mean()))
            epoch_ratio.append(float((ratio - 1.0).abs().max().detach()))
            epoch_grad.append(float(grad))
        def finite_mean(values: torch.Tensor) -> float:
            finite = values[torch.isfinite(values)]
            return float(finite.mean()) if finite.numel() else float("nan")

        positive_rows = torch.tensor(
            [pair.present for pair in batch], device=run_device, dtype=torch.bool
        )
        positive_empty = reward_components["empty_prediction"][positive_rows]
        logs.append({
            "update": update,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "dice_component_mean": finite_mean(reward_components["dice"]),
            "lesion_f1_component_mean": finite_mean(reward_components["lesion_f1"]),
            "boundary_f1_component_mean": finite_mean(reward_components["boundary_f1"]),
            "absent_reward_mean": finite_mean(reward_components["absent_reward"]),
            "empty_prediction_fraction": float(reward_components["empty_prediction"].mean()),
            "positive_empty_fraction": (
                float(positive_empty.mean()) if positive_empty.numel() else float("nan")
            ),
            "zero_advantage_fraction": float((spread <= 1e-6).float().mean()),
            "policy_loss": float(np.mean(epoch_loss)),
            "reference_kl": float(np.mean(epoch_kl)),
            "clip_fraction": float(np.mean(epoch_clip)),
            "max_abs_ratio_minus_one": float(np.max(epoch_ratio)),
            "grad_norm": float(np.mean(epoch_grad)),
        })
        if update % int(settings["log_interval"]) == 0 or update == 1:
            print(json.dumps({
                "dataset": dataset, "n": train_pair_count, "group": group_size,
                "kind": kind, **logs[-1],
            }), flush=True)

    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "training_log.csv", logs)
    payload = {
        "fingerprint": current_fingerprint,
        "state_dict": policy.state_dict(),
        "kind": kind,
        "dataset": dataset,
        "train_pair_count": int(train_pair_count),
        "group_size": int(group_size),
        "seed": seed,
        "method": cfg["method"],
        "embedding_dim": int(pairs[0].local.shape[-1]),
        "cnn_hidden_channels": int(cfg["policy"]["cnn_hidden_channels"]),
        "action_names": ACTION_NAMES,
        "reward": dict(cfg["reward"]),
        "reference_alpha": alpha_ref.cpu(),
        "reference_beta": beta_ref.cpu(),
        "checkpoint_selection": "final_update_no_validation",
        "controller_input": "none" if kind == "global" else "pre-SCM prompt-conditioned dense image representation",
    }
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    (destination / "metadata.json").write_text(json.dumps({
        key: value for key, value in payload.items()
        if key not in {"state_dict", "reference_alpha", "reference_beta"}
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def load_policy(path: Path, cfg: Mapping, device: torch.device) -> tuple[dict, nn.Module]:
    saved = torch.load(path, map_location=device)
    if saved.get("fingerprint") != fingerprint(cfg):
        raise RuntimeError(f"Checkpoint/config fingerprint mismatch: {path}")
    beta_floor = float(cfg["policy"]["beta_floor"])
    beta_max = float(cfg["policy"]["beta_max"])
    if saved["kind"] == "global":
        policy = GlobalBetaPolicy(4, beta_floor, beta_max)
    elif saved["kind"] == "cnn":
        policy = CNNBetaPolicy(
            int(saved["embedding_dim"]), int(saved["cnn_hidden_channels"]), 4, beta_floor, beta_max
        )
    else:
        raise ValueError(saved["kind"])
    policy.load_state_dict(saved["state_dict"])
    policy.to(device).eval()
    return saved, policy


def mean_actions_for_image(
    cfg: Mapping,
    dataset: str,
    item: Mapping,
    text_delta: torch.Tensor,
    device: torch.device,
    policy: nn.Module | None,
    kind: str,
    constant_action: torch.Tensor | None = None,
) -> torch.Tensor:
    classes = len(item["meta"]["class_names"])
    if kind == "fixed":
        return fixed_action(cfg, dataset, device)[None].expand(classes, -1)
    if kind == "constant":
        if constant_action is None:
            raise ValueError("constant action is required")
        return constant_action.to(device)[None].expand(classes, -1)
    if kind == "global":
        alpha, beta = policy.parameters_ab(classes)
    elif kind == "cnn":
        local = item["local"].float().to(device)[None].expand(classes, -1, -1)
        alpha, beta = policy.parameters_ab(
            local, text_delta.to(device), tuple(int(v) for v in item["meta"]["grid_shape"])
        )
    else:
        raise ValueError(kind)
    return map_actions(alpha / (alpha + beta), cfg)


def mean_actions_for_batch(
    cfg: Mapping,
    dataset: str,
    items: Sequence[Mapping],
    text_delta: torch.Tensor,
    device: torch.device,
    policy: nn.Module | None,
    kind: str,
    constant_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return deterministic mean actions as [images, classes, actions]."""
    batch = len(items)
    classes = len(items[0]["meta"]["class_names"])
    if kind == "fixed":
        return fixed_action(cfg, dataset, device)[None, None].expand(batch, classes, -1)
    if kind == "constant":
        if constant_action is None:
            raise ValueError("constant action is required")
        return constant_action.to(device)[None, None].expand(batch, classes, -1)
    if kind == "global":
        alpha, beta = policy.parameters_ab(batch * classes)
    elif kind == "cnn":
        grid_shape = tuple(int(v) for v in items[0]["meta"]["grid_shape"])
        local = torch.stack([item["local"].float() for item in items]).to(device)
        local = local[:, None].expand(-1, classes, -1, -1).reshape(
            batch * classes, local.shape[1], local.shape[2]
        )
        text = text_delta.to(device)[None].expand(batch, -1, -1).reshape(
            batch * classes, text_delta.shape[-1]
        )
        alpha, beta = policy.parameters_ab(local, text, grid_shape)
    else:
        raise ValueError(kind)
    return map_actions(alpha / (alpha + beta), cfg).reshape(batch, classes, -1)


def selected_image_rows(
    cfg: Mapping, train_dataset: str, train_pair_count: int, split: str
) -> list[dict]:
    if split == "internal":
        allowed = {row["image_id"] for row in read_csv(
            split_dir(cfg, train_dataset, train_pair_count) / "internal_images.csv"
        )}
        return [row for row in read_csv(cache_index(cfg, train_dataset)) if row["image_id"] in allowed]
    if split == "external":
        rows = []
        for dataset in cfg["datasets"]:
            if dataset != train_dataset:
                rows.extend(read_csv(cache_index(cfg, dataset)))
        return rows
    raise ValueError(split)


def summarize_per_pair(per_pair: Sequence[dict], datasets: Sequence[str]) -> list[dict]:
    summary = []
    for dataset in datasets:
        rows = [row for row in per_pair if row["dataset"] == dataset]
        if not rows:
            continue
        foreground = []
        for class_name in get_spec(dataset).foreground_classes:
            foreground.append(_nanmean(
                row["iou"] for row in rows if row["class_name"] == class_name
            ))
        background_by_image = {row["image_id"]: row["background_iou"] for row in rows}
        background = _nanmean(background_by_image.values())
        valid = [value for value in foreground if np.isfinite(value)]
        row = {
            "dataset": dataset,
            "num_images": len(background_by_image),
            "foreground_mIoU": _nanmean(foreground),
            "background_IoU": background,
            "mIoU": (sum(valid) + background) / (len(valid) + 1),
            "foreground_Dice": _nanmean(item["dice"] for item in rows),
            "absent_FP_area": _nanmean(item["absent_fp_area"] for item in rows),
        }
        for metric in ("foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "absent_FP_area"):
            row[f"{metric}_percent"] = 100.0 * row[metric] if np.isfinite(row[metric]) else float("nan")
        summary.append(row)
    if len(summary) > 1:
        macro = {"dataset": "macro_average", "num_images": sum(row["num_images"] for row in summary)}
        for metric in ("foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "absent_FP_area"):
            macro[metric] = _nanmean(row[metric] for row in summary)
            macro[f"{metric}_percent"] = 100.0 * macro[metric] if np.isfinite(macro[metric]) else float("nan")
        summary.append(macro)
    return summary


@torch.inference_mode()
def evaluate_provider(
    cfg: Mapping,
    train_dataset: str,
    train_pair_count: int,
    split: str,
    setting: str,
    device: str,
    policy: nn.Module | None = None,
    kind: str = "fixed",
    constant_action: torch.Tensor | None = None,
) -> tuple[list[dict], list[dict]]:
    run_device = torch.device(device)
    rows = selected_image_rows(cfg, train_dataset, train_pair_count, split)
    text_deltas = {dataset: load_text_delta(cfg, dataset) for dataset in cfg["datasets"]}
    per_pair = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    requested_batch = int(cfg["evaluation"].get("batch_size", 1))
    minimum_batch = int(cfg["evaluation"].get("min_batch_size", 1))
    evaluated = 0

    for dataset, dataset_rows in grouped.items():
        batch_size = max(minimum_batch, requested_batch)
        start = 0
        while start < len(dataset_rows):
            count = min(batch_size, len(dataset_rows) - start)
            chunk = dataset_rows[start:start + count]
            items = [torch.load(row["cache_path"], map_location="cpu") for row in chunk]
            try:
                classes = len(items[0]["meta"]["class_names"])
                grid_shape = tuple(int(v) for v in items[0]["meta"]["grid_shape"])
                actions = mean_actions_for_batch(
                    cfg, dataset, items, text_deltas[dataset], run_device,
                    policy, kind, constant_action,
                )
                hat = torch.stack([item["hat"].float() for item in items]).to(run_device)
                affinity = torch.stack([
                    item["base_affinity"].float() for item in items
                ]).to(run_device)
                coords = torch.stack([item["coords"].float() for item in items]).to(run_device)
                seeds = torch.stack([item["seed_idx"].long() for item in items]).to(run_device)
                seed_coords = torch.gather(
                    coords[:, None].expand(-1, classes, -1, -1), 2,
                    seeds[..., None, None].expand(-1, -1, 1, 2),
                ).squeeze(2)
                distance = ((coords[:, None] - seed_coords[:, :, None]) ** 2).sum(-1)
                eta, tau, gamma, kappa = actions.unbind(-1)
                adjusted_affinity = affinity * torch.exp(-kappa[..., None] * distance)
                refined = hat * (
                    1.0 - gamma[..., None] * (adjusted_affinity < tau[..., None]).to(hat.dtype)
                )
                target_size = tuple(int(v) for v in items[0]["gt_masks"].shape[-2:])
                scores = F.interpolate(
                    refined.reshape(count * classes, 1, *grid_shape),
                    size=target_size, mode="bilinear", align_corners=False,
                ).reshape(count, classes, *target_size)
                eligible = scores >= eta[..., None, None]
                any_eligible = eligible.any(1)
                winner = torch.where(eligible, scores, -torch.inf).argmax(1)
                class_ids = torch.arange(classes, device=run_device)[None, :, None, None]
                predicted = (
                    any_eligible[:, None] & (winner[:, None] == class_ids)
                ).cpu().numpy()
                gt_batch = torch.stack([item["gt_masks"].bool() for item in items]).numpy()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if "out of memory" not in str(error).lower() or count <= minimum_batch:
                    raise
                batch_size = max(minimum_batch, count // 2)
                del items
                gc.collect()
                torch.cuda.empty_cache()
                print(json.dumps({
                    "setting": setting, "split": split, "dataset": dataset,
                    "event": "evaluation_oom_retry", "batch_size": batch_size,
                }), flush=True)
                continue

            for item, predicted_image, gt_masks in zip(items, predicted, gt_batch):
                pred_any = predicted_image.any(0)
                gt_any = gt_masks.any(0)
                background = _safe_iou(~pred_any, ~gt_any)
                for class_idx, class_name in enumerate(item["meta"]["class_names"]):
                    pred, gt = predicted_image[class_idx], gt_masks[class_idx]
                    per_pair.append({
                        "dataset": dataset,
                        "image_id": item["meta"]["image_id"],
                        "class_name": class_name,
                        "gt_present": int(gt.any()),
                        "iou": _safe_iou(pred, gt),
                        "dice": _dice(pred, gt),
                        "absent_fp_area": float(pred.mean()) if not gt.any() else float("nan"),
                        "background_iou": background,
                    })
            start += count
            evaluated += count
            if evaluated % 500 < count or evaluated == len(rows):
                memory = torch.cuda.max_memory_allocated(run_device) / (1024 ** 3)
                print(json.dumps({
                    "setting": setting, "split": split, "evaluated": evaluated,
                    "total": len(rows), "dataset": dataset, "batch_size": count,
                    "max_gpu_GiB": round(memory, 2),
                }), flush=True)
    datasets = [train_dataset] if split == "internal" else [d for d in cfg["datasets"] if d != train_dataset]
    summary = summarize_per_pair(per_pair, datasets)
    for row in summary:
        row.update({
            "train_dataset": train_dataset,
            "train_pairs": int(train_pair_count),
            "split": split,
            "setting": setting,
        })
    return summary, per_pair


def evaluation_dir(
    cfg: Mapping, dataset: str, train_pairs: int, group_size: int | None, setting: str
) -> Path:
    group = "no_group" if group_size is None else f"g{group_size}"
    return output_root(cfg) / "evaluations" / dataset / f"n{train_pairs}" / group / setting


def evaluate_checkpoint(
    cfg: Mapping, checkpoint: Path, split: str, device: str, force: bool = False
) -> Path:
    saved, policy = load_policy(checkpoint, cfg, torch.device(device))
    root = evaluation_dir(
        cfg, saved["dataset"], saved["train_pair_count"], saved["group_size"], saved["kind"]
    )
    path = root / f"{split}_summary.csv"
    if path.is_file() and not force:
        existing = read_csv(path)
        if existing and existing[0].get("fingerprint") == fingerprint(cfg):
            return path
    summary, per_pair = evaluate_provider(
        cfg, saved["dataset"], saved["train_pair_count"], split, saved["kind"], device,
        policy=policy, kind=saved["kind"],
    )
    for row in summary:
        row["group_size"] = int(saved["group_size"])
        row["method"] = cfg["method"]
        row["fingerprint"] = fingerprint(cfg)
    write_csv(path, summary)
    if bool(cfg["evaluation"]["save_per_pair"]):
        write_csv(root / f"{split}_per_pair.csv", per_pair)
    return path


def evaluate_fixed(
    cfg: Mapping, train_dataset: str, train_pair_count: int, split: str, device: str, force: bool = False
) -> Path:
    root = evaluation_dir(cfg, train_dataset, train_pair_count, None, "fixed_ngsc")
    path = root / f"{split}_summary.csv"
    if path.is_file() and not force:
        existing = read_csv(path)
        if existing and existing[0].get("fingerprint") == fingerprint(cfg):
            return path
    summary, per_pair = evaluate_provider(
        cfg, train_dataset, train_pair_count, split, "fixed_ngsc", device, kind="fixed"
    )
    for row in summary:
        row["group_size"] = ""
        row["method"] = cfg["method"]
        row["fingerprint"] = fingerprint(cfg)
    write_csv(path, summary)
    if bool(cfg["evaluation"]["save_per_pair"]):
        write_csv(root / f"{split}_per_pair.csv", per_pair)
    return path


def grid_actions(cfg: Mapping) -> torch.Tensor:
    settings = cfg["upper_bound"]
    values = [settings[f"{name}_grid"] for name in ACTION_NAMES]
    return torch.tensor(list(itertools.product(*values)), dtype=torch.float32)
