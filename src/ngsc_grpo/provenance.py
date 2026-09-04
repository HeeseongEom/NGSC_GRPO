from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from .config import project_path
from .registry import DATASETS, prompt_inventory


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_project(cfg, verify_large_hashes: bool = False) -> dict:
    project_root = Path(cfg["_project_root"])
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    checkpoint = project_path(cfg, cfg["paths"]["checkpoint_dir"])
    required_checkpoint = [
        "config.json", "configuration_biomed_clip.py", "modeling_biomed_clip.py",
        "processing_biomed_clip.py", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "pytorch_model.bin",
    ]
    missing = [str(checkpoint / name) for name in required_checkpoint if not (checkpoint / name).is_file()]
    counts = {}
    for name, dataset in DATASETS.items():
        image_dir = data_root / dataset.directory / "test_images"
        mask_dir = data_root / dataset.directory / "test_masks"
        if not image_dir.is_dir():
            missing.append(str(image_dir))
        if not mask_dir.is_dir():
            missing.append(str(mask_dir))
        counts[name] = len([path for path in image_dir.iterdir() if path.is_file()]) if image_dir.is_dir() else 0

    template_path = project_path(cfg, cfg["paths"]["prompt_template"])
    spec = importlib.util.spec_from_file_location("ngsc_validate_templates", template_path)
    if spec is None or spec.loader is None:
        raise ImportError(template_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = module.BIOMEDCOOP_TEMPLATES
    prompt_lengths = {}
    for names in prompt_inventory(DATASETS).values():
        for name in names:
            prompt_lengths[name] = len(templates.get(name, []))
    for dataset in DATASETS.values():
        for key in (dataset.normal_class, dataset.normal_prompt_key, dataset.abnormal_prompt_key):
            prompt_lengths[key] = len(templates.get(key, []))
    short_prompts = {name: size for name, size in prompt_lengths.items() if size < 50}
    if short_prompts:
        missing.append(f"prompt classes with fewer than 50 templates: {short_prompts}")

    provenance_path = project_root / "assets" / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    hash_checks = {}
    for item in provenance.get("sha256", []):
        path = project_root / item["path"]
        if path.is_file() and (verify_large_hashes or path.stat().st_size < 100 * 1024 * 1024):
            actual = sha256(path)
            hash_checks[item["path"]] = actual == item["digest"]
            if actual != item["digest"]:
                missing.append(f"checksum mismatch: {path}")
    result = {
        "ok": not missing,
        "missing_or_invalid": missing,
        "dataset_image_counts": counts,
        "prompt_classes_checked": len(prompt_lengths),
        "hash_checks": hash_checks,
        "large_hashes_skipped": not verify_large_hashes,
    }
    if missing:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result
