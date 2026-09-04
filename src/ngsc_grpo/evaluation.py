from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .cache import cache_root, read_cache_index
from .config import ACTION_NAMES, BASELINES, config_fingerprint, experiment_root, project_path, require_method
from .core import (
    hard_masks_from_actions,
    original_image_normalize,
    standardize_state,
)
from .policy import LinearBetaController
from .registry import get_spec, load_class_masks
from .training import _device


def _safe_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if not bool(gt.any()):
        return float("nan")
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / max(1, union))


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    if not bool(gt.any()):
        return float("nan")
    return float(2 * np.logical_and(pred, gt).sum() / max(1, pred.sum() + gt.sum()))


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool).reshape(-1)
    scores = scores.astype(np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    # Exact Mann-Whitney rank sum with average ranks for ties.  Computing tie
    # groups in vectorized form avoids a Python loop over every image pixel.
    starts = np.concatenate((np.array([0]), np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1))
    ends = np.concatenate((starts[1:], np.array([scores.size])))
    positive_per_group = np.add.reduceat(labels[order].astype(np.int64), starts)
    average_ranks = 0.5 * (starts + 1 + ends)
    rank_sum = float(np.dot(positive_per_group, average_ranks))
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _nanmean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _assemble_multiclass(score_maps: np.ndarray, thresholds: np.ndarray) -> List[np.ndarray]:
    """Apply the design's eligible-set rule and return mutually exclusive class masks."""
    if score_maps.ndim != 3 or thresholds.shape != (score_maps.shape[0],):
        raise ValueError("score_maps must be [classes,H,W] and thresholds [classes]")
    eligible = score_maps >= thresholds[:, None, None]
    any_eligible = eligible.any(axis=0)
    winner = np.where(eligible, score_maps, -np.inf).argmax(axis=0)
    return [np.logical_and(any_eligible, winner == class_idx) for class_idx in range(score_maps.shape[0])]


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixed_action(cfg, radiology: bool) -> torch.Tensor:
    core = cfg["ngsc_core"]
    return torch.tensor(
        [
            core["radiology_eta"] if radiology else core["nonradiology_eta"],
            core["original_tau"],
            core["original_gamma"],
            0.0,
        ],
        dtype=torch.float32,
    )


class ActionProvider:
    def __init__(self, cfg, method: str, baseline: str, seed: int | None, device: torch.device):
        if baseline not in BASELINES:
            raise ValueError(f"Unknown baseline {baseline}; choices={BASELINES}")
        self.cfg = cfg
        self.method = method
        self.baseline = baseline
        self.seed = seed
        self.device = device
        self.policy = None
        self.state_mean = None
        self.state_std = None
        if baseline == "source_static":
            path = experiment_root(cfg) / method / "static_search" / "best_action.json"
            static = json.loads(path.read_text(encoding="utf-8"))
            self.static_action = torch.tensor(
                [static["action"][name] for name in ACTION_NAMES], dtype=torch.float32, device=device
            )
        elif baseline == "conditional_grpo":
            if seed is None:
                raise ValueError("conditional_grpo evaluation requires --seed")
            path = experiment_root(cfg) / method / "seeds" / f"seed_{seed}" / "policy_final.pt"
            saved = torch.load(path, map_location=device)
            if saved["config_fingerprint"] != config_fingerprint(cfg):
                raise RuntimeError(f"Policy/config fingerprint mismatch: {path}")
            controller = cfg["controller"]
            self.policy = LinearBetaController(
                state_dim=controller["input_dim"],
                action_dim=controller["action_dim"],
                beta_floor=controller["beta_floor"],
                beta_max=controller["beta_max"],
            ).to(device)
            self.policy.load_state_dict(saved["state_dict"])
            self.policy.eval()
            self.state_mean = saved["state_mean"].to(device)
            self.state_std = saved["state_std"].to(device)

    @torch.inference_mode()
    def __call__(self, state: torch.Tensor, radiology: bool):
        if self.baseline in {"original_ngsc", "core_fixed"}:
            action = _fixed_action(self.cfg, radiology).to(self.device)
            return action, None, None, state.to(self.device)
        if self.baseline == "source_static":
            return self.static_action, None, None, state.to(self.device)
        normalized_state = standardize_state(
            state.to(self.device), self.state_mean, self.state_std, self.cfg["state"]["clip"]
        )
        dist = self.policy.distribution(normalized_state)
        action = self.policy.mean_action(normalized_state, self.cfg["continuous_action"])
        return action, dist.concentration1, dist.concentration0, normalized_state


def _result_name(baseline: str, seed: int | None) -> str:
    return f"{baseline}_seed{seed}" if baseline == "conditional_grpo" else baseline


def evaluate_target(
    cfg,
    method: str,
    baseline: str,
    seed: int | None = None,
    device: str | None = None,
    force: bool = False,
) -> Path:
    require_method(method)
    name = _result_name(baseline, seed)
    result_root = experiment_root(cfg) / method / "results"
    result_path = result_root / f"{name}.csv"
    metadata_path = result_root / f"{name}_metadata.json"
    if result_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_fingerprint") == config_fingerprint(cfg):
            return result_path

    index_path = cache_root(cfg, method, "target") / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Target cache index is missing: {index_path}. Run cache-target first.")
    run_device = _device(cfg, device)
    provider = ActionProvider(cfg, method, baseline, seed, run_device)
    data_root = project_path(cfg, cfg["paths"]["data_root"])
    per_pair: List[dict] = []
    action_rows: List[dict] = []
    expected_fingerprint = config_fingerprint(cfg)

    with torch.inference_mode():
        for index_row in read_cache_index(index_path):
            payload = torch.load(index_row["cache_path"], map_location="cpu")
            meta = payload["meta"]
            if meta["config_fingerprint"] != expected_fingerprint:
                raise RuntimeError(f"Stale target cache: {index_row['cache_path']}")
            if "gt_masks" in payload or bool(meta.get("target_labels_read")):
                raise AssertionError("Target cache must not contain labels")
            spec = get_spec(meta["dataset"])
            masks = load_class_masks(data_root, meta, size=None)  # labels first enter here: evaluation only
            original_hat = original_image_normalize(payload["raw"].float())
            gt_masks = []
            score_maps = []
            thresholds = []
            for class_idx, class_name in enumerate(meta["class_names"]):
                state = payload["state"][class_idx].float()
                action, alpha, beta, normalized_state = provider(state, spec.radiology)
                hat = original_hat[class_idx] if baseline == "original_ngsc" else payload["hat"][class_idx].float()
                _, score = hard_masks_from_actions(
                    hat.to(run_device),
                    payload["base_affinity"][class_idx].float().to(run_device),
                    payload["coords"].float().to(run_device),
                    int(payload["seed_idx"][class_idx]),
                    action,
                    meta["grid_shape"],
                    meta["image_size"],
                )
                score_np = score.float().cpu().numpy()
                gt_np = masks[class_name].astype(bool)
                gt_masks.append(gt_np)
                score_maps.append(score_np)
                thresholds.append(float(action[0].cpu()))
                action_row = {
                    "dataset": meta["dataset"],
                    "image_id": meta["image_id"],
                    "class_name": class_name,
                }
                action_row.update({key: float(action[idx].cpu()) for idx, key in enumerate(ACTION_NAMES)})
                for idx in range(11):
                    action_row[f"state_{idx}"] = float(normalized_state[idx].cpu())
                for idx, key in enumerate(ACTION_NAMES):
                    action_row[f"alpha_{key}"] = float(alpha[idx].cpu()) if alpha is not None else float("nan")
                    action_row[f"beta_{key}"] = float(beta[idx].cpu()) if beta is not None else float("nan")
                action_rows.append(action_row)
            predicted_masks = _assemble_multiclass(
                np.stack(score_maps, axis=0), np.asarray(thresholds, dtype=np.float32)
            )
            pred_any = np.logical_or.reduce(predicted_masks)
            gt_any = np.logical_or.reduce(gt_masks)
            bg_iou = _safe_iou(~pred_any, ~gt_any)
            for class_idx, class_name in enumerate(meta["class_names"]):
                pred_np = predicted_masks[class_idx]
                gt_np = gt_masks[class_idx]
                per_pair.append(
                    {
                        "dataset": meta["dataset"],
                        "image_id": meta["image_id"],
                        "class_name": class_name,
                        "gt_present": int(gt_np.any()),
                        "iou": _safe_iou(pred_np, gt_np),
                        "dice": _dice(pred_np, gt_np),
                        "auroc": _binary_auroc(score_maps[class_idx], gt_np),
                        "absent_fp_area": float(pred_np.mean()) if not gt_np.any() else float("nan"),
                        "background_iou": bg_iou,
                    }
                )

    summary_rows: List[dict] = []
    for dataset in list(cfg["targets"]) + ["macro_average"]:
        if dataset == "macro_average":
            base_rows = summary_rows
            row = {
                "dataset": dataset,
                "foreground_mIoU": _nanmean(item["foreground_mIoU"] for item in base_rows),
                "background_IoU": _nanmean(item["background_IoU"] for item in base_rows),
                "mIoU": _nanmean(item["mIoU"] for item in base_rows),
                "foreground_Dice": _nanmean(item["foreground_Dice"] for item in base_rows),
                "AUROC": _nanmean(item["AUROC"] for item in base_rows),
                "absent_FP_area": _nanmean(item["absent_FP_area"] for item in base_rows),
                "num_images": int(sum(item["num_images"] for item in base_rows)),
                "num_present_pairs": int(sum(item["num_present_pairs"] for item in base_rows)),
            }
        else:
            rows = [item for item in per_pair if item["dataset"] == dataset]
            foreground_by_class = []
            for class_name in get_spec(dataset).foreground_classes:
                values = [item["iou"] for item in rows if item["class_name"] == class_name and np.isfinite(item["iou"])]
                foreground_by_class.append(float(np.mean(values)) if values else float("nan"))
            foreground_miou = _nanmean(foreground_by_class)
            bg_by_image: Dict[str, float] = {}
            for item in rows:
                bg_by_image[item["image_id"]] = item["background_iou"]
            background_iou = _nanmean(bg_by_image.values())
            valid_fg = [value for value in foreground_by_class if np.isfinite(value)]
            row = {
                "dataset": dataset,
                "foreground_mIoU": foreground_miou,
                "background_IoU": background_iou,
                "mIoU": float((sum(valid_fg) + background_iou) / (len(valid_fg) + 1)),
                "foreground_Dice": _nanmean(item["dice"] for item in rows),
                "AUROC": _nanmean(item["auroc"] for item in rows),
                "absent_FP_area": _nanmean(item["absent_fp_area"] for item in rows),
                "num_images": len(bg_by_image),
                "num_present_pairs": int(sum(item["gt_present"] for item in rows)),
            }
        summary_rows.append(row)

    for row in summary_rows:
        for metric in ("foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "AUROC", "absent_FP_area"):
            value = float(row[metric])
            row[f"{metric}_percent"] = 100.0 * value if np.isfinite(value) else float("nan")

    _write_csv(result_path, summary_rows)
    _write_csv(result_root / f"{name}_per_pair.csv", per_pair)
    diagnostic_root = experiment_root(cfg) / method / "diagnostics"
    _write_csv(diagnostic_root / f"{name}_action_rows.csv", action_rows)
    _write_action_stats(diagnostic_root / f"{name}_action_stats.csv", action_rows, cfg)
    metadata_path.write_text(
        json.dumps(
            {
                "method": method,
                "baseline": baseline,
                "seed": seed,
                "config_fingerprint": expected_fingerprint,
                "target_labels_used_for_training": False,
                "target_labels_used_for_search": False,
                "target_labels_read_stage": "evaluation_only",
                "target_adaptation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_path


def _write_action_stats(path: Path, rows: List[dict], cfg) -> None:
    output = []
    for dataset in cfg["targets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        states = np.asarray([[row[f"state_{idx}"] for idx in range(11)] for row in selected])
        for action_name in ACTION_NAMES:
            values = np.asarray([row[action_name] for row in selected], dtype=float)
            low, high = cfg["continuous_action"][action_name]
            width = high - low
            boundary = np.logical_or(values <= low + 0.01 * width, values >= high - 0.01 * width)
            correlations = []
            if values.std() > 1e-12:
                for idx in range(11):
                    correlations.append(
                        float(np.corrcoef(states[:, idx], values)[0, 1])
                        if states[:, idx].std() > 1e-12 else float("nan")
                    )
            else:
                correlations = [float("nan")] * 11
            alpha = np.asarray([row[f"alpha_{action_name}"] for row in selected], dtype=float)
            beta = np.asarray([row[f"beta_{action_name}"] for row in selected], dtype=float)
            output.append(
                {
                    "dataset": dataset,
                    "action": action_name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "boundary_hit_rate": float(boundary.mean()),
                    "beta_concentration_mean": float(np.nanmean(alpha + beta))
                    if np.isfinite(alpha + beta).any() else float("nan"),
                    "max_abs_state_correlation": float(np.nanmax(np.abs(correlations)))
                    if np.isfinite(correlations).any() else float("nan"),
                }
            )
    _write_csv(path, output)
