#!/usr/bin/env python3
"""Shared exp2 data, continuous-Beta policy, reward, training, and evaluation code."""

from __future__ import annotations

import copy
import csv
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
from PIL import Image
from torch import nn
from torch.distributions import Beta, kl_divergence

from ngsc_grpo.core import hard_masks_from_actions, upsample_patch_maps
from ngsc_grpo.evaluation import _assemble_multiclass, _binary_auroc, _dice, _nanmean, _safe_iou
from ngsc_grpo.registry import get_spec, load_class_masks


ROOT = Path(__file__).resolve().parents[2]
ALL_ACTIONS = ("eta", "tau", "gamma", "kappa_sp", "q")


def load_exp2_config(path: str | Path = ROOT / "configs" / "exp2.yaml") -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    return cfg


def exp1_root(cfg: Mapping) -> Path:
    return ROOT / cfg["experiment"]["output_root"] / cfg["experiment"]["exp1_name"]


def exp2_root(cfg: Mapping) -> Path:
    return ROOT / cfg["experiment"]["output_root"] / cfg["experiment"]["name"]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Pair:
    dataset: str
    role: str
    image_id: str
    image_relpath: str
    class_name: str
    present: bool
    hat: torch.Tensor
    seed_idx: int
    base_affinity: torch.Tensor
    coords: torch.Tensor
    state: torch.Tensor
    gt_mask: torch.Tensor
    grid_shape: tuple[int, int]

    @property
    def key(self) -> str:
        return "|".join((self.dataset, self.image_id, self.class_name))


def _cache_index(cfg: Mapping, method: str, split: str) -> Path:
    base = exp1_root(cfg)
    if split == "source":
        return base / method / "cache" / "source" / "index.csv"
    if split == "internal":
        return base / "internal_test" / method / "cache" / "index.csv"
    if split == "external":
        return base / method / "cache" / "target" / "index.csv"
    raise ValueError(split)


def load_source_pairs(cfg: Mapping, method: str) -> list[Pair]:
    pairs = []
    for row in read_csv(_cache_index(cfg, method, "source")):
        item = torch.load(row["cache_path"], map_location="cpu")
        meta = item["meta"]
        for idx, class_name in enumerate(meta["class_names"]):
            pairs.append(
                Pair(
                    dataset=meta["dataset"],
                    role=meta["role"],
                    image_id=meta["image_id"],
                    image_relpath=meta["image_relpath"],
                    class_name=class_name,
                    present=bool(item["present"][idx]),
                    hat=item["hat"][idx].float(),
                    seed_idx=int(item["seed_idx"][idx]),
                    base_affinity=item["base_affinity"][idx].float(),
                    coords=item["coords"].float(),
                    state=item["state"][idx].float(),
                    gt_mask=item["gt_masks"][idx].bool(),
                    grid_shape=tuple(int(v) for v in meta["grid_shape"]),
                )
            )
    return pairs


def load_prompt_state(cfg: Mapping, method: str) -> dict[str, float]:
    path = exp2_root(cfg) / "prompt_disagreement" / method / "values.csv"
    if not path.is_file():
        return {}
    return {row["key"]: float(row["prompt_js"]) for row in read_csv(path)}


def state_vector(pair: Pair, mode: str, prompt_state: Mapping[str, float]) -> torch.Tensor:
    if mode == "base11":
        return pair.state
    if mode == "prompt12":
        if pair.key not in prompt_state:
            raise KeyError(f"Missing prompt disagreement for {pair.key}")
        return torch.cat((pair.state, pair.state.new_tensor([prompt_state[pair.key]])))
    raise ValueError(mode)


class HierarchicalSampler:
    def __init__(self, pairs: Sequence[Pair], seed: int):
        self.rng = random.Random(seed)
        self.groups: dict[str, dict[str, dict[bool, list[Pair]]]] = {}
        for pair in pairs:
            self.groups.setdefault(pair.dataset, {}).setdefault(pair.class_name, {}).setdefault(
                pair.present, []
            ).append(pair)
        self.datasets = sorted(self.groups)

    def sample(self, count: int) -> list[Pair]:
        out = []
        for _ in range(int(count)):
            dataset = self.rng.choice(self.datasets)
            classes = self.groups[dataset]
            class_name = self.rng.choice(sorted(classes))
            presence = self.rng.choice(sorted(classes[class_name]))
            out.append(self.rng.choice(classes[class_name][presence]))
        return out


def action_names(cfg: Mapping, action_set: str) -> tuple[str, ...]:
    names = tuple(cfg["actions"]["sets"][action_set])
    if not names or any(name not in ALL_ACTIONS for name in names):
        raise ValueError(action_set)
    return names


def action_bounds(cfg: Mapping, names: Sequence[str], device=None, dtype=torch.float32):
    lows = torch.tensor([cfg["actions"][name][0] for name in names], device=device, dtype=dtype)
    highs = torch.tensor([cfg["actions"][name][1] for name in names], device=device, dtype=dtype)
    return lows, highs


def map_actions(z: torch.Tensor, cfg: Mapping, names: Sequence[str]) -> torch.Tensor:
    lows, highs = action_bounds(cfg, names, z.device, z.dtype)
    return lows + (highs - lows) * z


def action_column(actions: torch.Tensor, names: Sequence[str], name: str, default: float) -> torch.Tensor:
    if name in names:
        return actions[..., names.index(name)]
    return actions.new_full(actions.shape[:-1], float(default))


def source_static_action(cfg: Mapping, method: str, names: Sequence[str]) -> torch.Tensor:
    path = exp1_root(cfg) / method / "static_search" / "best_action.json"
    static = json.loads(path.read_text(encoding="utf-8"))["action"]
    defaults = {"q": 0.95}
    return torch.tensor([static.get(name, defaults[name]) if name in defaults else static[name] for name in names])


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def reference_ab(
    cfg: Mapping, reference_action: torch.Tensor, names: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lows, highs = action_bounds(cfg, names, device=device)
    mean = ((reference_action.to(device) - lows) / (highs - lows)).clamp(0.05, 0.95)
    concentration = float(cfg["policy"]["reference_concentration"])
    return mean * concentration, (1.0 - mean) * concentration


class GlobalBetaPolicy(nn.Module):
    def __init__(self, action_dim: int, beta_floor: float, beta_max: float):
        super().__init__()
        self.action_dim = int(action_dim)
        self.beta_floor = float(beta_floor)
        self.beta_max = float(beta_max)
        self.raw = nn.Parameter(torch.zeros(action_dim, 2))

    def initialize(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        target = torch.stack((alpha, beta), dim=-1) - self.beta_floor
        self.raw.data.copy_(inverse_softplus(target.clamp_min(1e-6)))

    def parameters_ab(self, state: torch.Tensor | None, batch: int | None = None):
        ab = (self.beta_floor + F.softplus(self.raw)).clamp(max=self.beta_max)
        if state is not None:
            ab = ab.expand(state.shape[0], -1, -1)
        elif batch is not None:
            ab = ab.expand(int(batch), -1, -1)
        return ab[..., 0], ab[..., 1]


class ConditionalBetaPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, beta_floor: float, beta_max: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.beta_floor = float(beta_floor)
        self.beta_max = float(beta_max)
        self.linear = nn.Linear(state_dim, action_dim * 2)

    def initialize(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        target = torch.stack((alpha, beta), dim=-1) - self.beta_floor
        self.linear.weight.data.zero_()
        self.linear.bias.data.copy_(inverse_softplus(target.clamp_min(1e-6)).reshape(-1))

    def parameters_ab(self, state: torch.Tensor, batch: int | None = None):
        raw = self.linear(state).reshape(state.shape[0], self.action_dim, 2)
        ab = (self.beta_floor + F.softplus(raw)).clamp(max=self.beta_max)
        return ab[..., 0], ab[..., 1]


def distribution(policy: nn.Module, state: torch.Tensor | None, batch: int | None = None) -> Beta:
    alpha, beta = policy.parameters_ab(state, batch=batch)
    return Beta(alpha, beta)


def standardized_states(
    pairs: Sequence[Pair], mode: str, prompt_state: Mapping[str, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.stack([state_vector(pair, mode, prompt_state) for pair in pairs])
    mean = values.mean(0)
    std = values.std(0, unbiased=False).clamp_min(1e-6)
    return values, mean, std


def normalize_state(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, cfg: Mapping):
    low, high = cfg["policy"]["state_clip"]
    return ((value - mean) / (std + 1e-6)).clamp(float(low), float(high))


def evidence(hat: torch.Tensor, rho: float) -> torch.Tensor:
    return torch.logsumexp(float(rho) * hat.float(), dim=-1) / float(rho) - math.log(hat.shape[-1]) / float(rho)


def null_evidence(pairs: Sequence[Pair], rho: float) -> torch.Tensor:
    values = [evidence(pair.hat, rho) for pair in pairs if not pair.present]
    if len(values) < 8:
        raise RuntimeError(f"At least 8 absent source pairs are required for q, found {len(values)}")
    return torch.sort(torch.stack(values).float()).values


def quantile_from_sorted(values: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    values = values.to(q.device)
    position = q.clamp(0, 1) * (values.numel() - 1)
    lower = position.floor().long()
    upper = position.ceil().long()
    weight = position - lower.float()
    return values[lower] * (1.0 - weight) + values[upper] * weight


def refined_patch_scores(
    pair: Pair,
    actions: torch.Tensor,
    names: Sequence[str],
    soft_affinity: bool,
    temperature_tau: float,
) -> torch.Tensor:
    tau = action_column(actions, names, "tau", 0.6)
    gamma = action_column(actions, names, "gamma", 0.5)
    kappa = action_column(actions, names, "kappa_sp", 0.0)
    coords = pair.coords.to(actions.device)
    dist2 = ((coords - coords[pair.seed_idx]) ** 2).sum(-1)
    affinity = pair.base_affinity.to(actions.device).unsqueeze(0) * torch.exp(
        -kappa[:, None] * dist2[None]
    )
    if soft_affinity:
        suppressed = torch.sigmoid((tau[:, None] - affinity) / float(temperature_tau))
    else:
        suppressed = (affinity < tau[:, None]).to(actions.dtype)
    return pair.hat.to(actions.device).unsqueeze(0) * (1.0 - gamma[:, None] * suppressed)


def reward_actions(
    pair: Pair,
    actions: torch.Tensor,
    names: Sequence[str],
    reward_name: str,
    cfg: Mapping,
    null_values: torch.Tensor | None,
) -> torch.Tensor:
    settings = cfg["reward"]
    soft = reward_name != "hard"
    refined = refined_patch_scores(
        pair,
        actions,
        names,
        soft_affinity=soft,
        temperature_tau=float(settings["temperature_tau"]),
    )
    score = upsample_patch_maps(refined, pair.grid_shape, pair.gt_mask.shape[-2:])
    eta = action_column(actions, names, "eta", 1.4)
    gt = pair.gt_mask.to(actions.device).float().unsqueeze(0)

    if soft:
        prediction = torch.sigmoid((score - eta[:, None, None]) / float(settings["temperature_eta"]))
    else:
        prediction = (score >= eta[:, None, None]).float()

    gate = prediction.new_ones(prediction.shape[0])
    if "q" in names:
        if null_values is None:
            raise ValueError("q action requires null evidence values")
        q = action_column(actions, names, "q", 0.95)
        delta = quantile_from_sorted(null_values, q)
        pair_evidence = evidence(pair.hat.to(actions.device), float(settings["rho"]))
        if soft:
            gate = torch.sigmoid(
                (pair_evidence - delta) / float(settings["temperature_evidence"])
            )
        else:
            gate = (pair_evidence >= delta).to(prediction.dtype)
        prediction = prediction * gate[:, None, None]

    if reward_name == "hard":
        if pair.present:
            inter = (prediction * gt).sum((-2, -1))
            return 2.0 * inter / (prediction.sum((-2, -1)) + gt.sum()).clamp_min(1e-8)
        return 1.0 - prediction.mean((-2, -1))

    if pair.present:
        inter = (prediction * gt).sum((-2, -1))
        union = prediction.sum((-2, -1)) + gt.sum() - inter
        soft_iou = inter / union.clamp_min(1e-8)
        tpr = inter / gt.sum().clamp_min(1e-8)
        negative = 1.0 - gt
        fpr = (prediction * negative).sum((-2, -1)) / negative.sum().clamp_min(1e-8)
        youden = 0.5 * (tpr - fpr + 1.0)
        if reward_name == "soft_iou":
            pixel_reward = soft_iou
        else:
            pixel_reward = (
                float(settings["soft_iou_weight"]) * soft_iou
                + float(settings["soft_youden_weight"]) * youden
            )
    else:
        pixel_reward = 1.0 - prediction.mean((-2, -1))

    if reward_name in {"soft_iou", "soft_iou_youden"}:
        return pixel_reward
    if reward_name == "soft_full":
        target_presence = prediction.new_full(gate.shape, float(pair.present))
        presence_reward = 1.0 - (gate - target_presence).square()
        return (
            float(settings["pixel_weight_with_presence"]) * pixel_reward
            + float(settings["presence_weight"]) * presence_reward
        )
    raise ValueError(reward_name)


def policy_mean_action(
    policy: nn.Module,
    cfg: Mapping,
    names: Sequence[str],
    state: torch.Tensor | None,
    batch: int | None = None,
) -> torch.Tensor:
    dist = distribution(policy, state, batch=batch)
    return map_actions(dist.mean, cfg, names)


def _pair_action(
    pair: Pair,
    policy: nn.Module,
    kind: str,
    cfg: Mapping,
    names: Sequence[str],
    device: torch.device,
    state_mode: str,
    prompt_state: Mapping[str, float],
    state_mean: torch.Tensor | None,
    state_std: torch.Tensor | None,
) -> torch.Tensor:
    if kind == "global":
        return policy_mean_action(policy, cfg, names, None, batch=1)[0]
    value = state_vector(pair, state_mode, prompt_state).to(device)
    value = normalize_state(value, state_mean, state_std, cfg).unsqueeze(0)
    return policy_mean_action(policy, cfg, names, value)[0]


def validation_metrics(
    pairs: Sequence[Pair],
    policy: nn.Module,
    kind: str,
    cfg: Mapping,
    names: Sequence[str],
    device: torch.device,
    state_mode: str,
    prompt_state: Mapping[str, float],
    state_mean: torch.Tensor | None,
    state_std: torch.Tensor | None,
    null_values: torch.Tensor | None,
    reward_name: str,
) -> tuple[float, float]:
    rewards = []
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.dataset, pair.image_id)].append(pair)
    per_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for pair in pairs:
            action = _pair_action(
                pair, policy, kind, cfg, names, device, state_mode, prompt_state, state_mean, state_std
            )
            rewards.append(
                float(reward_actions(pair, action.unsqueeze(0), names, reward_name, cfg, null_values)[0])
            )
        for (dataset, _), image_pairs in grouped.items():
            scores, thresholds, gt_masks, present_gate = [], [], [], []
            for pair in image_pairs:
                action = _pair_action(
                    pair, policy, kind, cfg, names, device, state_mode, prompt_state, state_mean, state_std
                )
                refined = refined_patch_scores(pair, action.unsqueeze(0), names, False, 1.0)[0]
                score = upsample_patch_maps(refined, pair.grid_shape, pair.gt_mask.shape[-2:])
                scores.append(score.cpu().numpy())
                thresholds.append(float(action[names.index("eta")]) if "eta" in names else 1.4)
                gt_masks.append(pair.gt_mask.numpy())
                allowed = True
                if "q" in names:
                    q = action[names.index("q")].reshape(1)
                    delta = quantile_from_sorted(null_values, q)[0]
                    allowed = bool(evidence(pair.hat.to(device), float(cfg["reward"]["rho"])) >= delta)
                present_gate.append(allowed)
            thresholds_np = np.asarray(thresholds, dtype=np.float32)
            thresholds_np[~np.asarray(present_gate)] = np.inf
            predicted = _assemble_multiclass(np.stack(scores), thresholds_np)
            pred_any = np.logical_or.reduce(predicted)
            gt_any = np.logical_or.reduce(gt_masks)
            per_dataset[dataset]["background"].append(_safe_iou(~pred_any, ~gt_any))
            for pair, pred, gt in zip(image_pairs, predicted, gt_masks):
                value = _safe_iou(pred, gt)
                if np.isfinite(value):
                    per_dataset[dataset][pair.class_name].append(value)
    dataset_mious = []
    for dataset, values in per_dataset.items():
        class_iou = [_nanmean(values[name]) for name in get_spec(dataset).foreground_classes]
        bg = _nanmean(values["background"])
        valid = [value for value in class_iou if np.isfinite(value)]
        dataset_mious.append((sum(valid) + bg) / (len(valid) + 1))
    return float(np.mean(rewards)), float(np.mean(dataset_mious))


def train_policy(
    cfg: Mapping,
    method: str,
    run_name: str,
    kind: str,
    action_set: str,
    reward_name: str,
    seed: int,
    device: str,
    optimization: str,
    heldout_domain: str | None = None,
    state_mode: str = "base11",
    force: bool = False,
) -> Path:
    names = action_names(cfg, action_set)
    output = exp2_root(cfg) / "runs" / run_name / method / f"seed_{seed}"
    checkpoint_path = output / "policy_best.pt"
    if checkpoint_path.is_file() and not force:
        return checkpoint_path
    run_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    all_pairs = load_source_pairs(cfg, method)
    if heldout_domain:
        train_pairs = [p for p in all_pairs if p.role == "source_train" and p.dataset != heldout_domain]
        val_pairs = [p for p in all_pairs if p.dataset == heldout_domain]
    else:
        train_pairs = [p for p in all_pairs if p.role == "source_train"]
        val_pairs = [p for p in all_pairs if p.role == "source_val"]
    prompt_state = load_prompt_state(cfg, method) if state_mode == "prompt12" else {}

    state_mean = state_std = None
    if kind == "conditional":
        _, state_mean, state_std = standardized_states(train_pairs, state_mode, prompt_state)
        state_mean, state_std = state_mean.to(run_device), state_std.to(run_device)
        state_dim = int(state_mean.numel())
    else:
        state_dim = 0

    reference_action = source_static_action(cfg, method, names).to(run_device)
    ref_alpha, ref_beta = reference_ab(cfg, reference_action, names, run_device)
    policy_args = (
        len(names), float(cfg["policy"]["beta_floor"]), float(cfg["policy"]["beta_max"])
    )
    if kind == "global":
        policy = GlobalBetaPolicy(*policy_args).to(run_device)
    elif kind == "conditional":
        policy = ConditionalBetaPolicy(state_dim, *policy_args).to(run_device)
    else:
        raise ValueError(kind)
    policy.initialize(ref_alpha, ref_beta)

    settings = cfg[optimization]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["lr"]))
    sampler = HierarchicalSampler(train_pairs, seed + 10000)
    null_values = null_evidence(train_pairs, float(cfg["reward"]["rho"])).to(run_device) if "q" in names else None
    logs = []
    best_score = -float("inf")
    best_state = None
    best_update = 0

    for update in range(1, int(settings["updates"]) + 1):
        batch = sampler.sample(int(settings["batch_states"]))
        batch_size = len(batch)
        if kind == "conditional":
            state = torch.stack([state_vector(p, state_mode, prompt_state) for p in batch]).to(run_device)
            state = normalize_state(state, state_mean, state_std, cfg)
        else:
            state = None

        with torch.no_grad():
            old_dist = distribution(policy, state, batch=batch_size)
            old_alpha = old_dist.concentration1.detach().clone()
            old_beta = old_dist.concentration0.detach().clone()
            frozen_old = Beta(old_alpha, old_beta)
            normalized = frozen_old.sample((int(settings["group_size"]),)).permute(1, 0, 2).contiguous()
            old_log_prob = frozen_old.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
            physical = map_actions(normalized, cfg, names)
            rewards = torch.stack(
                [reward_actions(pair, physical[idx], names, reward_name, cfg, null_values) for idx, pair in enumerate(batch)]
            )
            centered = rewards - rewards.mean(1, keepdim=True)
            spread = rewards.std(1, keepdim=True, unbiased=False)
            advantage = torch.where(spread > 1e-6, centered / (spread + 1e-6), torch.zeros_like(centered))

        epoch_losses, epoch_clip, epoch_ratio_dev, epoch_kl, epoch_grad = [], [], [], [], []
        for _ in range(int(settings["ppo_epochs"])):
            current = distribution(policy, state, batch=batch_size)
            log_prob = current.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
            ratio = torch.exp(log_prob - old_log_prob)
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - float(settings["clip_epsilon"]), 1.0 + float(settings["clip_epsilon"])
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            ref = Beta(ref_alpha, ref_beta)
            reference_kl = kl_divergence(current, ref).sum(-1).mean()
            loss = policy_loss + float(settings["kl_beta"]) * reference_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(policy_loss.detach()))
            epoch_clip.append(float(((ratio < 1.0 - float(settings["clip_epsilon"])) | (ratio > 1.0 + float(settings["clip_epsilon"]))).float().mean()))
            epoch_ratio_dev.append(float((ratio - 1.0).abs().max().detach()))
            epoch_kl.append(float(reference_kl.detach()))
            epoch_grad.append(float(grad))

        row = {
            "update": update,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_zero_fraction": float((spread <= 1e-6).float().mean()),
            "policy_loss": float(np.mean(epoch_losses)),
            "clip_fraction": float(np.mean(epoch_clip)),
            "max_abs_ratio_minus_one": float(np.max(epoch_ratio_dev)),
            "reference_kl": float(np.mean(epoch_kl)),
            "grad_norm": float(np.mean(epoch_grad)),
            "val_reward": "",
            "val_mIoU": "",
        }
        if update == 1 or update % int(settings["val_interval"]) == 0 or update == int(settings["updates"]):
            val_reward, val_miou = validation_metrics(
                val_pairs, policy, kind, cfg, names, run_device, state_mode, prompt_state,
                state_mean, state_std, null_values, reward_name
            )
            row["val_reward"] = val_reward
            row["val_mIoU"] = val_miou
            if val_miou > best_score:
                best_score = val_miou
                best_update = update
                best_state = copy.deepcopy(policy.state_dict())
        logs.append(row)

    if best_state is None:
        raise AssertionError("No validation checkpoint selected")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_log.csv", logs)
    payload = {
        "state_dict": best_state,
        "kind": kind,
        "method": method,
        "run_name": run_name,
        "action_set": action_set,
        "action_names": names,
        "reward_name": reward_name,
        "seed": seed,
        "heldout_domain": heldout_domain,
        "state_mode": state_mode,
        "state_mean": None if state_mean is None else state_mean.cpu(),
        "state_std": None if state_std is None else state_std.cpu(),
        "reference_action": reference_action.cpu(),
        "reference_alpha": ref_alpha.cpu(),
        "reference_beta": ref_beta.cpu(),
        "null_evidence": None if null_values is None else null_values.cpu(),
        "best_update": best_update,
        "best_val_mIoU": best_score,
        "optimization": optimization,
    }
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    (output / "run_metadata.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k not in {"state_dict", "state_mean", "state_std", "reference_alpha", "reference_beta", "null_evidence"} and not torch.is_tensor(v)}, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def load_policy_checkpoint(path: Path, cfg: Mapping, device: torch.device):
    saved = torch.load(path, map_location=device)
    names = tuple(saved["action_names"])
    if saved["kind"] == "global":
        policy = GlobalBetaPolicy(
            len(names), float(cfg["policy"]["beta_floor"]), float(cfg["policy"]["beta_max"])
        ).to(device)
    else:
        state_dim = int(saved["state_mean"].numel())
        policy = ConditionalBetaPolicy(
            state_dim, len(names), float(cfg["policy"]["beta_floor"]), float(cfg["policy"]["beta_max"])
        ).to(device)
    policy.load_state_dict(saved["state_dict"])
    policy.eval()
    return saved, policy


def evaluate_checkpoint(
    cfg: Mapping,
    checkpoint: Path,
    split: str,
    device: str,
    force: bool = False,
) -> Path:
    run_device = torch.device(device)
    saved, policy = load_policy_checkpoint(checkpoint, cfg, run_device)
    method = saved["method"]
    result_root = exp2_root(cfg) / "evaluations" / saved["run_name"] / method / f"seed_{saved['seed']}"
    result_path = result_root / f"{split}_summary.csv"
    if result_path.is_file() and not force:
        return result_path
    prompt_state = load_prompt_state(cfg, method) if saved["state_mode"] == "prompt12" else {}
    names = tuple(saved["action_names"])
    null_values = saved["null_evidence"]
    if null_values is not None:
        null_values = null_values.to(run_device)
    state_mean = saved["state_mean"]
    state_std = saved["state_std"]
    if state_mean is not None:
        state_mean, state_std = state_mean.to(run_device), state_std.to(run_device)
    data_root = ROOT / "data"
    per_pair, action_rows = [], []
    datasets = cfg["sources"] if split == "internal" else cfg["targets"]

    for index_row in read_csv(_cache_index(cfg, method, split)):
        item = torch.load(index_row["cache_path"], map_location="cpu")
        meta = item["meta"]
        masks = load_class_masks(data_root, meta, size=None)
        scores, thresholds, gt_masks = [], [], []
        for idx, class_name in enumerate(meta["class_names"]):
            pair = Pair(
                dataset=meta["dataset"], role=meta["role"], image_id=meta["image_id"],
                image_relpath=meta["image_relpath"], class_name=class_name,
                present=bool(masks[class_name].any()), hat=item["hat"][idx].float(),
                seed_idx=int(item["seed_idx"][idx]), base_affinity=item["base_affinity"][idx].float(),
                coords=item["coords"].float(), state=item["state"][idx].float(),
                gt_mask=torch.from_numpy(masks[class_name]), grid_shape=tuple(meta["grid_shape"]),
            )
            action = _pair_action(
                pair, policy, saved["kind"], cfg, names, run_device, saved["state_mode"],
                prompt_state, state_mean, state_std
            ).detach()
            refined = refined_patch_scores(pair, action.unsqueeze(0), names, False, 1.0)[0]
            score = upsample_patch_maps(refined, pair.grid_shape, meta["image_size"])
            threshold = float(action[names.index("eta")]) if "eta" in names else 1.4
            allowed = True
            if "q" in names:
                q = action[names.index("q")].reshape(1)
                delta = quantile_from_sorted(null_values, q)[0]
                allowed = bool(evidence(pair.hat.to(run_device), float(cfg["reward"]["rho"])) >= delta)
            scores.append(score.cpu().numpy())
            thresholds.append(threshold if allowed else float("inf"))
            gt_masks.append(masks[class_name])
            action_rows.append({
                "dataset": meta["dataset"], "image_id": meta["image_id"], "class_name": class_name,
                **{name: float(action[j]) for j, name in enumerate(names)}, "presence_allowed": int(allowed),
            })
        predicted = _assemble_multiclass(np.stack(scores), np.asarray(thresholds, dtype=np.float32))
        pred_any = np.logical_or.reduce(predicted)
        gt_any = np.logical_or.reduce(gt_masks)
        background_iou = _safe_iou(~pred_any, ~gt_any)
        for idx, class_name in enumerate(meta["class_names"]):
            pred, gt = predicted[idx], gt_masks[idx]
            per_pair.append({
                "dataset": meta["dataset"], "image_id": meta["image_id"], "class_name": class_name,
                "gt_present": int(gt.any()), "iou": _safe_iou(pred, gt), "dice": _dice(pred, gt),
                "auroc": _binary_auroc(scores[idx], gt),
                "absent_fp_area": float(pred.mean()) if not gt.any() else float("nan"),
                "background_iou": background_iou,
            })

    summary = []
    for dataset in list(datasets) + ["macro_average"]:
        if dataset == "macro_average":
            bases = summary
            row = {metric: _nanmean(item[metric] for item in bases) for metric in (
                "foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "AUROC", "absent_FP_area"
            )}
            row.update({"dataset": dataset, "num_images": sum(item["num_images"] for item in bases)})
        else:
            rows = [item for item in per_pair if item["dataset"] == dataset]
            foreground = []
            for class_name in get_spec(dataset).foreground_classes:
                foreground.append(_nanmean(item["iou"] for item in rows if item["class_name"] == class_name))
            bg_by_image = {item["image_id"]: item["background_iou"] for item in rows}
            bg = _nanmean(bg_by_image.values())
            valid = [value for value in foreground if np.isfinite(value)]
            row = {
                "dataset": dataset, "foreground_mIoU": _nanmean(foreground), "background_IoU": bg,
                "mIoU": (sum(valid) + bg) / (len(valid) + 1),
                "foreground_Dice": _nanmean(item["dice"] for item in rows),
                "AUROC": _nanmean(item["auroc"] for item in rows),
                "absent_FP_area": _nanmean(item["absent_fp_area"] for item in rows),
                "num_images": len(bg_by_image),
            }
        for metric in ("foreground_mIoU", "background_IoU", "mIoU", "foreground_Dice", "AUROC", "absent_FP_area"):
            row[f"{metric}_percent"] = 100.0 * row[metric] if np.isfinite(row[metric]) else float("nan")
        summary.append(row)
    write_csv(result_path, summary)
    write_csv(result_root / f"{split}_per_pair.csv", per_pair)
    write_csv(result_root / f"{split}_actions.csv", action_rows)
    return result_path
