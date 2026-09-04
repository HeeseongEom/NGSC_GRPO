from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Beta, kl_divergence
import torch.nn.functional as F

from .core import map_normalized_actions, unmap_physical_actions


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


@dataclass(frozen=True)
class ReferenceBeta:
    alpha: torch.Tensor
    beta: torch.Tensor

    def distribution(self, device=None, dtype=None) -> Beta:
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        return Beta(alpha, beta)


def reference_from_action(
    physical_action: torch.Tensor,
    bounds: Mapping[str, Sequence[float]],
    concentration: float = 20.0,
) -> ReferenceBeta:
    mean = unmap_physical_actions(physical_action.float(), bounds).clamp(0.05, 0.95)
    return ReferenceBeta(mean * concentration, (1.0 - mean) * concentration)


class LinearBetaController(nn.Module):
    def __init__(self, state_dim=11, action_dim=4, beta_floor=0.5, beta_max=100.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.beta_floor = float(beta_floor)
        self.beta_max = float(beta_max)
        self.linear = nn.Linear(self.state_dim, self.action_dim * 2)

    def parameters_ab(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.linear(state).reshape(*state.shape[:-1], self.action_dim, 2)
        ab = self.beta_floor + F.softplus(raw)
        ab = ab.clamp(max=self.beta_max)
        return ab[..., 0], ab[..., 1]

    def distribution(self, state: torch.Tensor) -> Beta:
        alpha, beta = self.parameters_ab(state)
        return Beta(alpha, beta)

    @torch.no_grad()
    def initialize_as_reference(self, reference: ReferenceBeta) -> None:
        target_a = (reference.alpha - self.beta_floor).clamp_min(1e-6)
        target_b = (reference.beta - self.beta_floor).clamp_min(1e-6)
        raw = torch.stack((inverse_softplus(target_a), inverse_softplus(target_b)), dim=-1)
        self.linear.weight.zero_()
        self.linear.bias.copy_(raw.reshape(-1).to(self.linear.bias))

    def mean_action(self, state: torch.Tensor, bounds) -> torch.Tensor:
        dist = self.distribution(state)
        return map_normalized_actions(dist.mean, bounds)


def analytic_reference_kl(dist: Beta, reference: ReferenceBeta) -> torch.Tensor:
    ref = reference.distribution(device=dist.concentration1.device, dtype=dist.concentration1.dtype)
    return kl_divergence(dist, ref).sum(dim=-1)
