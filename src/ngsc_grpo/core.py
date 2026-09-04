from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ACTION_NAMES = ("eta", "tau", "gamma", "kappa_sp")


def per_class_normalize(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each class over its patches. Input shape is [..., classes, patches]."""
    mean = raw.mean(dim=-1, keepdim=True)
    std = raw.std(dim=-1, keepdim=True, unbiased=False)
    return (raw - mean) / (std + eps)


def original_image_normalize(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Legacy NGSC normalization over all foreground classes and patches in one image."""
    mean = raw.mean()
    # The legacy implementation used torch.std() without overriding Bessel correction.
    std = raw.std(unbiased=True)
    return (raw - mean) / (std + eps)


def normalized_coords(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def map_normalized_actions(
    z: torch.Tensor, bounds: Mapping[str, Sequence[float]]
) -> torch.Tensor:
    if z.shape[-1] != 4:
        raise ValueError(f"Expected action dimension 4, got {tuple(z.shape)}")
    lows = z.new_tensor([bounds[name][0] for name in ACTION_NAMES])
    highs = z.new_tensor([bounds[name][1] for name in ACTION_NAMES])
    return lows + (highs - lows) * z


def unmap_physical_actions(
    actions: torch.Tensor, bounds: Mapping[str, Sequence[float]]
) -> torch.Tensor:
    lows = actions.new_tensor([bounds[name][0] for name in ACTION_NAMES])
    highs = actions.new_tensor([bounds[name][1] for name in ACTION_NAMES])
    return (actions - lows) / (highs - lows)


def apply_continuous_ngsc(
    hat_lambda: torch.Tensor,
    base_affinity: torch.Tensor,
    coords: torch.Tensor,
    seed_idx: int,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply continuous spatial affinity and PAF for one image-class state.

    actions may be [4] or [G, 4]. Returned refined maps and affinities have
    shape [N] or [G, N], respectively.
    """
    squeeze = actions.ndim == 1
    if squeeze:
        actions = actions.unsqueeze(0)
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError(f"actions must be [4] or [G,4], got {tuple(actions.shape)}")
    if hat_lambda.ndim != 1 or base_affinity.shape != hat_lambda.shape:
        raise ValueError("hat_lambda and base_affinity must be matching 1-D patch vectors")
    if coords.shape != (hat_lambda.numel(), 2):
        raise ValueError("coords must have shape [patches,2]")
    if not 0 <= int(seed_idx) < hat_lambda.numel():
        raise IndexError("seed_idx is out of range")

    eta, tau, gamma, kappa_sp = actions.unbind(dim=-1)
    del eta
    dist2 = ((coords - coords[int(seed_idx)]) ** 2).sum(dim=-1)
    affinity = base_affinity.unsqueeze(0) * torch.exp(-kappa_sp[:, None] * dist2[None])
    low_affinity = affinity < tau[:, None]
    factor = 1.0 - gamma[:, None] * low_affinity.to(hat_lambda.dtype)
    refined = hat_lambda.unsqueeze(0) * factor
    if squeeze:
        return refined[0], affinity[0]
    return refined, affinity


def upsample_patch_maps(maps: torch.Tensor, grid_shape: Sequence[int], size: Sequence[int]) -> torch.Tensor:
    squeeze = maps.ndim == 1
    if squeeze:
        maps = maps.unsqueeze(0)
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    if maps.shape[-1] != gh * gw:
        raise ValueError("Patch count does not match grid_shape")
    up = F.interpolate(maps.reshape(-1, 1, gh, gw), size=tuple(size), mode="bilinear", align_corners=False)
    up = up[:, 0]
    return up[0] if squeeze else up


def hard_masks_from_actions(
    hat_lambda: torch.Tensor,
    base_affinity: torch.Tensor,
    coords: torch.Tensor,
    seed_idx: int,
    actions: torch.Tensor,
    grid_shape: Sequence[int],
    output_size: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    refined, _ = apply_continuous_ngsc(hat_lambda, base_affinity, coords, seed_idx, actions)
    up = upsample_patch_maps(refined, grid_shape, output_size)
    eta = actions[..., 0]
    if actions.ndim == 1:
        masks = up >= eta
    else:
        masks = up >= eta[:, None, None]
    return masks, up


def reward_for_actions(
    hat_lambda: torch.Tensor,
    base_affinity: torch.Tensor,
    coords: torch.Tensor,
    seed_idx: int,
    actions: torch.Tensor,
    grid_shape: Sequence[int],
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    masks, _ = hard_masks_from_actions(
        hat_lambda, base_affinity, coords, seed_idx, actions, grid_shape, gt_mask.shape[-2:]
    )
    gt = gt_mask.bool().unsqueeze(0)
    if bool(gt_mask.any()):
        inter = (masks & gt).sum(dim=(-2, -1)).float()
        denom = masks.sum(dim=(-2, -1)).float() + gt.sum().float()
        return 2.0 * inter / denom.clamp_min(1e-8)
    return 1.0 - masks.float().mean(dim=(-2, -1))


def extract_state(raw: torch.Tensor, hat: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
    if raw.ndim != 1 or raw.shape != hat.shape or raw.shape != affinity.shape:
        raise ValueError("raw, hat, and affinity must be matching 1-D tensors")
    n = raw.numel()
    topk = max(1, int((0.10 * n) + 0.999999))
    probs = torch.softmax(hat.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    entropy = entropy / torch.log(hat.new_tensor(float(n))).clamp_min(1e-12)
    return torch.stack(
        (
            raw.mean(),
            raw.std(unbiased=False),
            torch.quantile(raw, 0.90),
            torch.quantile(raw, 0.99),
            raw.max(),
            hat.max(),
            torch.topk(hat, topk).values.mean(),
            (hat > 0).float().mean(),
            entropy.to(hat.dtype),
            affinity.mean(),
            torch.quantile(affinity, 0.90),
        )
    ).float()


def fit_state_standardizer(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or states.shape[1] != 11:
        raise ValueError("states must have shape [N,11]")
    return states.mean(dim=0), states.std(dim=0, unbiased=False).clamp_min(1e-6)


def standardize_state(
    state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, clip_range=(-5.0, 5.0)
) -> torch.Tensor:
    return ((state - mean) / (std + 1e-6)).clamp(float(clip_range[0]), float(clip_range[1]))
