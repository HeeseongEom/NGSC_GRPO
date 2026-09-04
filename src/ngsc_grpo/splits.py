from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .config import config_fingerprint, experiment_root, project_path
from .registry import discover_records, normalize_class_name, prompt_inventory


FIELDS = (
    "dataset", "role", "modality", "image_id", "patient_id", "image_relpath",
    "present_classes", "prompt_classes", "patient_strategy",
)


def _primary_stratum(record: Mapping) -> str:
    present = record["present_classes"]
    return present[0] if present else "__empty__"


def stratified_take(records: Sequence[dict], count: int, seed: int) -> tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[_primary_stratum(record)].append(record)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected: List[dict] = []
    while len(selected) < min(count, len(records)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    selected_ids = {id(item) for item in selected}
    remaining = [item for item in records if id(item) not in selected_ids]
    return selected, remaining


def _write_manifest(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["present_classes"] = json.dumps(payload["present_classes"], ensure_ascii=False)
            payload["prompt_classes"] = json.dumps(payload["prompt_classes"], ensure_ascii=False)
            writer.writerow({field: payload[field] for field in FIELDS})


def read_manifest(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["present_classes"] = json.loads(row["present_classes"])
            row["prompt_classes"] = json.loads(row["prompt_classes"])
            rows.append(row)
    return rows


def assert_no_leakage(source_train: Sequence[dict], source_val: Sequence[dict], targets: Sequence[dict], cfg) -> None:
    source_names = {row["dataset"] for row in source_train + source_val}
    target_names = {row["dataset"] for row in targets}
    if source_names & target_names:
        raise AssertionError("Source and target dataset names overlap")
    source_paths = {row["image_relpath"] for row in source_train + source_val}
    target_paths = {row["image_relpath"] for row in targets}
    if source_paths & target_paths:
        raise AssertionError("Target samples occur in source manifests")
    train_units = {(row["dataset"], row["patient_id"]) for row in source_train}
    val_units = {(row["dataset"], row["patient_id"]) for row in source_val}
    if train_units & val_units:
        raise AssertionError("Patient/image unit leakage between source train and validation")

    source_classes = {
        normalize_class_name(name)
        for names in prompt_inventory(cfg["sources"]).values() for name in names
    }
    target_classes = {
        normalize_class_name(name)
        for names in prompt_inventory(cfg["targets"]).values() for name in names
    }
    overlap = source_classes & target_classes
    if overlap:
        raise AssertionError(f"Exact normalized source/target class overlap: {sorted(overlap)}")
    if any(row["present_classes"] for row in targets):
        raise AssertionError("Target manifest must not contain labels before evaluation")


def build_splits(cfg, force: bool = False) -> Path:
    output_dir = experiment_root(cfg) / "splits"
    train_path = output_dir / "source_train_manifest.csv"
    val_path = output_dir / "source_val_manifest.csv"
    target_path = output_dir / "target_manifest.csv"
    metadata_path = output_dir / "split_metadata.json"
    fingerprint = config_fingerprint(cfg)
    if not force and all(path.is_file() for path in (train_path, val_path, target_path, metadata_path)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_fingerprint") == fingerprint:
            assert_no_leakage(read_manifest(train_path), read_manifest(val_path), read_manifest(target_path), cfg)
            return output_dir

    data_root = project_path(cfg, cfg["paths"]["data_root"])
    prompt_map_dir = project_path(cfg, cfg["paths"]["prompt_map_dir"])
    split_seed = int(cfg["experiment"]["split_seed"])
    cal = cfg["calibration"]
    source_train: List[dict] = []
    source_val: List[dict] = []
    for offset, dataset_name in enumerate(cfg["sources"]):
        records = discover_records(data_root, prompt_map_dir, dataset_name, include_labels=True)
        selected, _ = stratified_take(records, int(cal["max_per_source"]), split_seed + offset * 101)
        train, remaining = stratified_take(selected, int(cal["train_per_source"]), split_seed + offset * 101 + 1)
        val, _ = stratified_take(remaining, int(cal["val_per_source"]), split_seed + offset * 101 + 2)
        for row in train:
            row["role"] = "source_train"
        for row in val:
            row["role"] = "source_val"
        source_train.extend(train)
        source_val.extend(val)

    targets: List[dict] = []
    for dataset_name in cfg["targets"]:
        rows = discover_records(data_root, prompt_map_dir, dataset_name, include_labels=False)
        for row in rows:
            row["role"] = "target"
        targets.extend(rows)

    assert_no_leakage(source_train, source_val, targets, cfg)
    _write_manifest(train_path, source_train)
    _write_manifest(val_path, source_val)
    _write_manifest(target_path, targets)
    inventory = {
        "source_prompts": prompt_inventory(cfg["sources"]),
        "target_prompts": prompt_inventory(cfg["targets"]),
        "assertions": {
            "dataset_disjoint": True,
            "sample_disjoint": True,
            "patient_unit_disjoint": True,
            "exact_normalized_class_disjoint": True,
            "target_labels_materialized_in_manifest": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "leakage_report.json").open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, ensure_ascii=False, indent=2)
    metadata_path.write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "split_seed": split_seed,
                "source_train_rows": len(source_train),
                "source_val_rows": len(source_val),
                "target_rows": len(targets),
                "target_labels_materialized": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def print_prompt_inventory(cfg) -> None:
    print("[LEAKAGE] source prompt classes")
    for dataset, names in prompt_inventory(cfg["sources"]).items():
        print(f"  {dataset}: {names}")
    print("[LEAKAGE] held-out target prompt classes")
    for dataset, names in prompt_inventory(cfg["targets"]).items():
        print(f"  {dataset}: {names}")
