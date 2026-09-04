from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml

from .cache import cache_root, read_cache_index
from .config import ACTION_NAMES, config_fingerprint, experiment_root, require_method
from .core import fit_state_standardizer, map_normalized_actions, reward_for_actions, standardize_state
from .policy import LinearBetaController, analytic_reference_kl, reference_from_action


@dataclass
class CachedPair:
    dataset: str
    role: str
    image_id: str
    class_name: str
    present: bool
    raw: torch.Tensor
    hat: torch.Tensor
    seed_idx: int
    base_affinity: torch.Tensor
    coords: torch.Tensor
    state: torch.Tensor
    gt_mask: torch.Tensor
    grid_shape: tuple[int, int]


def _device(cfg, requested: str | None = None) -> torch.device:
    value = requested or cfg["runtime"]["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_source_pairs(cfg, method: str) -> List[CachedPair]:
    index_path = cache_root(cfg, method, "source") / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Source cache index is missing: {index_path}. Run cache-source first.")
    pairs: List[CachedPair] = []
    expected_fingerprint = config_fingerprint(cfg)
    for row in read_cache_index(index_path):
        item = torch.load(row["cache_path"], map_location="cpu")
        meta = item["meta"]
        if meta["config_fingerprint"] != expected_fingerprint or meta["method"] != method:
            raise RuntimeError(f"Stale or mismatched cache: {row['cache_path']}")
        if "gt_masks" not in item:
            raise RuntimeError(f"Source cache has no reward masks: {row['cache_path']}")
        for class_idx, class_name in enumerate(meta["class_names"]):
            pairs.append(
                CachedPair(
                    dataset=meta["dataset"],
                    role=meta["role"],
                    image_id=meta["image_id"],
                    class_name=class_name,
                    present=bool(item["present"][class_idx]),
                    raw=item["raw"][class_idx].float(),
                    hat=item["hat"][class_idx].float(),
                    seed_idx=int(item["seed_idx"][class_idx]),
                    base_affinity=item["base_affinity"][class_idx].float(),
                    coords=item["coords"].float(),
                    state=item["state"][class_idx].float(),
                    gt_mask=item["gt_masks"][class_idx].bool(),
                    grid_shape=(int(meta["grid_shape"][0]), int(meta["grid_shape"][1])),
                )
            )
    if not pairs:
        raise RuntimeError("The source cache contains no image-class pairs")
    return pairs


class HierarchicalPairSampler:
    """Uniform dataset -> class -> positive/negative sampling with deterministic RNG."""

    def __init__(self, pairs: Sequence[CachedPair], seed: int):
        self.rng = random.Random(seed)
        self.groups: Dict[str, Dict[str, Dict[bool, List[CachedPair]]]] = {}
        for pair in pairs:
            self.groups.setdefault(pair.dataset, {}).setdefault(pair.class_name, {}).setdefault(
                pair.present, []
            ).append(pair)
        self.datasets = sorted(self.groups)

    def one(self) -> CachedPair:
        dataset = self.rng.choice(self.datasets)
        classes = self.groups[dataset]
        class_name = self.rng.choice(sorted(classes))
        presence_groups = classes[class_name]
        presence = self.rng.choice(sorted(presence_groups))
        return self.rng.choice(presence_groups[presence])

    def sample(self, count: int) -> List[CachedPair]:
        return [self.one() for _ in range(int(count))]


def _reward(pair: CachedPair, actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    return reward_for_actions(
        pair.hat.to(device),
        pair.base_affinity.to(device),
        pair.coords.to(device),
        pair.seed_idx,
        actions,
        pair.grid_shape,
        pair.gt_mask.to(device),
    )


def _write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("Cannot infer CSV fields from an empty row list")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _static_paths(cfg, method: str) -> tuple[Path, Path]:
    root = experiment_root(cfg) / method / "static_search"
    return root / "best_action.json", root / "candidates.csv"


def run_static_search(cfg, method: str, device: str | None = None, force: bool = False) -> dict:
    require_method(method)
    best_path, candidates_path = _static_paths(cfg, method)
    fingerprint = config_fingerprint(cfg)
    if best_path.is_file() and not force:
        result = json.loads(best_path.read_text(encoding="utf-8"))
        if result.get("config_fingerprint") == fingerprint:
            return result

    all_pairs = load_source_pairs(cfg, method)
    train_pairs = [pair for pair in all_pairs if pair.role == "source_train"]
    settings = cfg["static_search"]
    search_seed = int(cfg["experiment"]["split_seed"]) + 101
    sampled = HierarchicalPairSampler(train_pairs, search_seed).sample(
        int(settings["num_states"])
    )
    count = int(settings["num_actions"])
    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=search_seed)
    normalized = sobol.draw(count)
    actions = map_normalized_actions(normalized, cfg["continuous_action"])
    run_device = _device(cfg, device)
    reward_sum = torch.zeros(count, dtype=torch.float64)
    batch_size = int(settings.get("action_chunk_size", 32))

    with torch.inference_mode():
        for pair in sampled:
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                values = _reward(pair, actions[start:end].to(run_device), run_device)
                reward_sum[start:end] += values.double().cpu()
    mean_reward = reward_sum / float(len(sampled))
    best_idx = int(mean_reward.argmax())
    best_action = actions[best_idx]
    rows = []
    for idx in range(count):
        row = {"candidate": idx, "mean_reward": float(mean_reward[idx])}
        row.update({name: float(actions[idx, j]) for j, name in enumerate(ACTION_NAMES)})
        rows.append(row)
    _write_csv(candidates_path, rows)
    _write_csv(
        candidates_path.with_name("rewards.csv"),
        [{"candidate": row["candidate"], "mean_reward": row["mean_reward"]} for row in rows],
    )
    result = {
        "method": method,
        "config_fingerprint": fingerprint,
        "num_candidates": count,
        "num_sampled_pairs": len(sampled),
        "candidate_index": best_idx,
        "mean_reward": float(mean_reward[best_idx]),
        "action": {name: float(best_action[j]) for j, name in enumerate(ACTION_NAMES)},
    }
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _deterministic_reward(
    pairs: Sequence[CachedPair],
    policy: LinearBetaController,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    bounds,
    device: torch.device,
) -> float:
    if not pairs:
        return float("nan")
    values = []
    with torch.inference_mode():
        for pair in pairs:
            state = standardize_state(pair.state.to(device), state_mean, state_std)
            action = policy.mean_action(state, bounds)
            values.append(float(_reward(pair, action, device)[0].cpu()))
    return float(np.mean(values))


def train_grpo(
    cfg,
    method: str,
    seed: int,
    device: str | None = None,
    force: bool = False,
) -> Path:
    require_method(method)
    output = experiment_root(cfg) / method / "seeds" / f"seed_{seed}"
    checkpoint_path = output / "policy_final.pt"
    fingerprint = config_fingerprint(cfg)
    if checkpoint_path.is_file() and not force:
        existing = torch.load(checkpoint_path, map_location="cpu")
        if existing.get("config_fingerprint") == fingerprint:
            return checkpoint_path

    _seed_everything(int(seed))
    run_device = _device(cfg, device)
    pairs = load_source_pairs(cfg, method)
    train_pairs = [pair for pair in pairs if pair.role == "source_train"]
    val_pairs = [pair for pair in pairs if pair.role == "source_val"]
    if not train_pairs or not val_pairs:
        raise RuntimeError("Both source_train and source_val cache pairs are required")
    states = torch.stack([pair.state for pair in train_pairs], dim=0)
    state_mean, state_std = fit_state_standardizer(states)
    state_mean, state_std = state_mean.to(run_device), state_std.to(run_device)

    static = run_static_search(cfg, method, device=str(run_device), force=False)
    reference_action = torch.tensor(
        [static["action"][name] for name in ACTION_NAMES], dtype=torch.float32, device=run_device
    )
    reference = reference_from_action(
        reference_action, cfg["continuous_action"], float(cfg["static_search"]["reference_concentration"])
    )
    policy = LinearBetaController(
        state_dim=int(cfg["state"]["dim"]),
        action_dim=4,
        beta_floor=float(cfg["controller"]["beta_floor"]),
        beta_max=float(cfg["controller"]["beta_max"]),
    ).to(run_device)
    policy.initialize_as_reference(reference)
    if sum(parameter.numel() for parameter in policy.parameters()) != 96:
        raise AssertionError("The feasibility controller must contain exactly 96 parameters")

    settings = cfg["grpo"]
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"])
    )
    sampler = HierarchicalPairSampler(train_pairs, int(seed) + 10_000)
    updates = int(settings["updates"])
    batch_size = int(settings["batch_states"])
    group_size = int(settings["group_size"])
    clip_eps = float(settings["clip_epsilon"])
    kl_coef = float(settings["kl_beta"])
    adv_eps = float(settings["advantage_eps"])
    val_interval = int(settings["val_interval"])
    logs: List[dict] = []

    for update in range(1, updates + 1):
        batch = sampler.sample(batch_size)
        state_batch = torch.stack([pair.state for pair in batch], dim=0).to(run_device)
        state_batch = standardize_state(state_batch, state_mean, state_std)
        old_dist = policy.distribution(state_batch)
        with torch.no_grad():
            normalized_action = old_dist.sample((group_size,)).transpose(0, 1).contiguous()
            old_log_prob = old_dist.log_prob(normalized_action.transpose(0, 1)).sum(-1).transpose(0, 1)
            physical_action = map_normalized_actions(normalized_action, cfg["continuous_action"])
            reward = torch.stack(
                [_reward(pair, physical_action[idx], run_device) for idx, pair in enumerate(batch)], dim=0
            )
            centered = reward - reward.mean(dim=1, keepdim=True)
            spread = reward.std(dim=1, keepdim=True, unbiased=False)
            advantage = torch.where(spread > adv_eps, centered / (spread + adv_eps), torch.zeros_like(centered))

        new_dist = policy.distribution(state_batch)
        new_log_prob = new_dist.log_prob(normalized_action.transpose(0, 1)).sum(-1).transpose(0, 1)
        ratio = torch.exp(new_log_prob - old_log_prob)
        objective_a = ratio * advantage
        objective_b = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy_loss = -torch.minimum(objective_a, objective_b).mean()
        reference_kl = analytic_reference_kl(new_dist, reference).mean()
        loss = policy_loss + kl_coef * reference_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip"]))
        optimizer.step()

        row = {
            "update": update,
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "reference_kl": float(reference_kl.detach()),
            "reward_mean": float(reward.mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "advantage_zero_fraction": float((spread <= adv_eps).float().mean()),
            "grad_norm": float(torch.as_tensor(grad_norm)),
            "source_val_reward": "",
        }
        with torch.no_grad():
            alpha_now, beta_now = new_dist.concentration1, new_dist.concentration0
            concentration_now = alpha_now + beta_now
            for action_idx, action_name in enumerate(ACTION_NAMES):
                row[f"action_{action_name}_mean"] = float(physical_action[..., action_idx].mean())
                row[f"action_{action_name}_std"] = float(
                    physical_action[..., action_idx].std(unbiased=False)
                )
                row[f"beta_concentration_{action_name}_mean"] = float(
                    concentration_now[..., action_idx].mean()
                )
                row[f"beta_concentration_{action_name}_std"] = float(
                    concentration_now[..., action_idx].std(unbiased=False)
                )
        if update == 1 or update % val_interval == 0 or update == updates:
            row["source_val_reward"] = _deterministic_reward(
                val_pairs, policy, state_mean, state_std, cfg["continuous_action"], run_device
            )
        logs.append(row)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "training_log.csv", logs)
    log_root = output / "logs"
    _write_csv(
        log_root / "train_reward.csv",
        [
            {"update": row["update"], "reward_mean": row["reward_mean"], "reward_std": row["reward_std"]}
            for row in logs
        ],
    )
    _write_csv(
        log_root / "source_val_reward.csv",
        [
            {"update": row["update"], "source_val_reward": row["source_val_reward"]}
            for row in logs if row["source_val_reward"] != ""
        ],
    )
    _write_csv(
        log_root / "kl.csv",
        [{"update": row["update"], "reference_kl": row["reference_kl"]} for row in logs],
    )
    _write_csv(
        log_root / "action_stats.csv",
        [
            {
                "update": row["update"],
                **{
                    key: row[key]
                    for key in row if key.startswith("action_")
                },
            }
            for row in logs
        ],
    )
    _write_csv(
        log_root / "beta_stats.csv",
        [
            {
                "update": row["update"],
                **{
                    key: row[key]
                    for key in row if key.startswith("beta_concentration_")
                },
            }
            for row in logs
        ],
    )
    (output / "config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False),
        encoding="utf-8",
    )
    np.save(output / "state_mean.npy", state_mean.detach().cpu().numpy())
    np.save(output / "state_std.npy", state_std.detach().cpu().numpy())
    payload = {
        "state_dict": policy.state_dict(),
        "state_mean": state_mean.detach().cpu(),
        "state_std": state_std.detach().cpu(),
        "reference_action": reference_action.detach().cpu(),
        "reference_alpha": reference.alpha.detach().cpu(),
        "reference_beta": reference.beta.detach().cpu(),
        "method": method,
        "seed": int(seed),
        "config_fingerprint": fingerprint,
        "controller_parameters": 96,
        "final_source_val_reward": logs[-1]["source_val_reward"],
    }
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    (output / "reference_beta.json").write_text(
        json.dumps(
            {
                "physical_action": {name: float(reference_action[idx].cpu()) for idx, name in enumerate(ACTION_NAMES)},
                "alpha": reference.alpha.cpu().tolist(),
                "beta": reference.beta.cpu().tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": method,
                "seed": int(seed),
                "config_fingerprint": fingerprint,
                "device": str(run_device),
                "updates": updates,
                "batch_size": batch_size,
                "group_size": group_size,
                "reference_action": static["action"],
                "target_labels_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path
