from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


METHODS = ("MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP")
BASELINES = ("original_ngsc", "core_fixed", "source_static", "conditional_grpo")
ACTION_NAMES = ("eta", "tau", "gamma", "kappa_sp")


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    if cfg["experiment"]["target_mode"] not in {"zero_shot", "one_normal_shot"}:
        raise ValueError("target_mode must be zero_shot or one_normal_shot")
    if cfg["experiment"]["target_mode"] != "zero_shot":
        raise NotImplementedError(
            "one_normal_shot is an intentional API hook only; a disjoint genuine-normal manifest is required"
        )
    if set(cfg["sources"]) & set(cfg["targets"]):
        raise ValueError("Source and target dataset names must be disjoint")
    if tuple(cfg["dense_methods"]) != METHODS:
        missing = set(METHODS) - set(cfg["dense_methods"])
        if missing:
            raise ValueError(f"All four dense CLIP methods are required; missing={sorted(missing)}")
    if cfg["state"]["dim"] != 11:
        raise ValueError("The feasibility state is fixed to 11 dimensions")
    for name in ACTION_NAMES:
        bounds = cfg["continuous_action"][name]
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"Invalid action bounds for {name}: {bounds}")
    cal = cfg["calibration"]
    if cal["train_per_source"] + cal["val_per_source"] > cal["max_per_source"]:
        raise ValueError("train_per_source + val_per_source exceeds max_per_source")


def project_path(cfg: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg["_project_root"]) / path
    return path.resolve()


def experiment_root(cfg: Dict[str, Any]) -> Path:
    return project_path(cfg, cfg["experiment"]["output_root"]) / cfg["experiment"]["name"]


def method_root(cfg: Dict[str, Any], method: str) -> Path:
    require_method(method)
    return experiment_root(cfg) / method


def seed_root(cfg: Dict[str, Any], method: str, seed: int) -> Path:
    return method_root(cfg, method) / "seeds" / f"seed_{seed}"


def require_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    project_root = Path(cfg["_project_root"])
    implementation_files = sorted((project_root / "src" / "ngsc_grpo").glob("*.py"))
    implementation_files.extend(
        path for path in (
            project_path(cfg, cfg["paths"]["prompt_template"]),
            project_root / "assets" / "PROVENANCE.json",
        ) if path.is_file()
    )
    prompt_map_dir = project_path(cfg, cfg["paths"]["prompt_map_dir"])
    implementation_files.extend(sorted(prompt_map_dir.glob("*.json")))
    for path in implementation_files:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
