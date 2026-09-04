from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import config_fingerprint, experiment_root, project_path, require_method
from .core import extract_state
from .registry import get_spec, load_class_masks
from .splits import read_manifest


INDEX_FIELDS = ("dataset", "role", "image_id", "image_relpath", "cache_path")


def cache_root(cfg, method: str, role: str) -> Path:
    return experiment_root(cfg) / method / "cache" / role


def _cache_name(row) -> str:
    digest = hashlib.sha1(row["image_relpath"].encode("utf-8")).hexdigest()[:12]
    return f"{Path(row['image_id']).stem}_{digest}.pt"


def _cast_cache_tensor(value: torch.Tensor, cfg) -> torch.Tensor:
    dtype = torch.float16 if cfg["runtime"]["cache_dtype"] == "float16" else torch.float32
    return value.detach().to(device="cpu", dtype=dtype)


def _manifest_for_role(cfg, role: str) -> List[dict]:
    split_dir = experiment_root(cfg) / "splits"
    if role == "source":
        return read_manifest(split_dir / "source_train_manifest.csv") + read_manifest(
            split_dir / "source_val_manifest.csv"
        )
    if role == "target":
        rows = read_manifest(split_dir / "target_manifest.csv")
        limit = cfg["runtime"].get("target_limit_per_dataset")
        if limit is not None:
            kept = []
            counts = {}
            for row in rows:
                counts.setdefault(row["dataset"], 0)
                if counts[row["dataset"]] < int(limit):
                    kept.append(row)
                    counts[row["dataset"]] += 1
            return kept
        return rows
    raise ValueError("role must be source or target")


def build_feature_cache(cfg, method: str, role: str, device: str | None = None, force: bool = False) -> Path:
    require_method(method)
    rows = _manifest_for_role(cfg, role)
    root = cache_root(cfg, method, role)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(cfg)
    extractor = None
    index_rows = []
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    reward_size = int(cfg["calibration"]["reward_resolution"])

    for row in tqdm(rows, desc=f"cache:{method}:{role}"):
        dataset_dir = root / row["dataset"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / _cache_name(row)
        valid_existing = False
        if path.is_file() and not force:
            try:
                existing = torch.load(path, map_location="cpu")
                valid_existing = (
                    existing["meta"]["method"] == method
                    and existing["meta"]["config_fingerprint"] == fingerprint
                    and existing["meta"]["image_relpath"] == row["image_relpath"]
                )
            except Exception:
                valid_existing = False
        if not valid_existing:
            if extractor is None:
                from .model_adapter import DenseBiomedCLIP
                extractor = DenseBiomedCLIP(cfg, method, device=device)
            image = Image.open(data_root / row["image_relpath"]).convert("RGB")
            frozen = extractor.frozen_ngsc_quantities(image, get_spec(row["dataset"]))
            class_names = list(get_spec(row["dataset"]).foreground_classes)
            states = torch.stack(
                [
                    extract_state(frozen["raw"][idx], frozen["hat"][idx], frozen["base_affinity"][idx])
                    for idx in range(len(class_names))
                ],
                dim=0,
            )
            payload = {
                "meta": {
                    "method": method,
                    "role": row["role"],
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "image_relpath": row["image_relpath"],
                    "patient_id": row["patient_id"],
                    "class_names": class_names,
                    "grid_shape": list(frozen["grid_shape"]),
                    "image_size": [image.height, image.width],
                    "config_fingerprint": fingerprint,
                    "target_labels_read": False,
                },
                "raw": _cast_cache_tensor(frozen["raw"], cfg),
                "hat": _cast_cache_tensor(frozen["hat"], cfg),
                "seed_idx": frozen["seed_idx"].detach().cpu().long(),
                "base_affinity": _cast_cache_tensor(frozen["base_affinity"], cfg),
                "coords": _cast_cache_tensor(frozen["coords"], cfg),
                "state": states.cpu().float(),
            }
            if role == "source":
                masks = load_class_masks(data_root, row, size=(reward_size, reward_size))
                gt_masks = np.stack([masks[name] for name in class_names], axis=0)
                payload["gt_masks"] = torch.from_numpy(gt_masks).bool()
                payload["present"] = payload["gt_masks"].flatten(1).any(dim=1)
                payload["meta"]["target_labels_read"] = False
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        index_rows.append(
            {
                "dataset": row["dataset"],
                "role": row["role"],
                "image_id": row["image_id"],
                "image_relpath": row["image_relpath"],
                "cache_path": str(path.resolve()),
            }
        )

    index_path = root / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    metadata = {
        "method": method,
        "role": role,
        "num_images": len(index_rows),
        "config_fingerprint": fingerprint,
        "target_labels_read": False,
    }
    with (root / "cache_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return index_path


def read_cache_index(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
