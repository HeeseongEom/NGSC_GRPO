import torch

from ngsc_grpo.core import (
    apply_continuous_ngsc,
    extract_state,
    map_normalized_actions,
    per_class_normalize,
    reward_for_actions,
)


BOUNDS = {"eta": [0.0, 3.0], "tau": [0.0, 1.0], "gamma": [0.0, 1.0], "kappa_sp": [0.0, 4.0]}


def test_per_class_normalization_is_independent():
    raw = torch.tensor([[1.0, 2.0, 3.0], [100.0, 102.0, 104.0]])
    normalized = per_class_normalize(raw)
    assert torch.allclose(normalized.mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.std(-1, unbiased=False), torch.ones(2), atol=1e-5)


def test_action_mapping_and_continuous_endpoints():
    mapped = map_normalized_actions(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0] * 4]), BOUNDS)
    assert torch.equal(mapped[0], torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert torch.equal(mapped[1], torch.tensor([3.0, 1.0, 1.0, 4.0]))
    hat = torch.tensor([-1.0, 0.5, 2.0, 1.0])
    affinity = torch.tensor([0.0, 0.2, 0.8, 1.0])
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    unchanged, _ = apply_continuous_ngsc(hat, affinity, coords, 2, torch.tensor([1.0, 0.5, 0.0, 3.0]))
    assert torch.equal(unchanged, hat)
    suppressed, _ = apply_continuous_ngsc(hat, affinity, coords, 2, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(suppressed[:3], torch.zeros(3))
    assert suppressed[3] == hat[3]


def test_positive_and_negative_reward_contract():
    hat = torch.tensor([2.0, -1.0, -1.0, -1.0])
    affinity = torch.ones(4)
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gt = torch.tensor([[True, False], [False, False]])
    assert torch.allclose(reward_for_actions(hat, affinity, coords, 0, action, (2, 2), gt), torch.ones(1))
    empty = torch.zeros((2, 2), dtype=torch.bool)
    assert torch.allclose(reward_for_actions(hat, affinity, coords, 0, action, (2, 2), empty), torch.tensor([0.75]))


def test_state_contract_is_finite_and_11d():
    raw = torch.linspace(-2, 3, 196)
    hat = per_class_normalize(raw.unsqueeze(0))[0]
    affinity = torch.linspace(0, 1, 196)
    state = extract_state(raw, hat, affinity)
    assert state.shape == (11,)
    assert torch.isfinite(state).all()
