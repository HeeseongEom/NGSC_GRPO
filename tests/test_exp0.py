from pathlib import Path
import sys

import torch


EXP0 = Path(__file__).resolve().parents[1] / "scripts" / "exp0"
sys.path.insert(0, str(EXP0))

from build_matrix import build_rows  # noqa: E402
from common import (  # noqa: E402
    CNNBetaPolicy,
    Pair,
    _balanced_prompt_sample,
    _quota,
    hard_reward,
    load_config,
    map_actions,
    reference_beta,
)


def test_exp0_matrix_has_56_jobs_and_96_policies():
    cfg = load_config()
    rows = build_rows(cfg)
    assert len(rows) == 56
    assert sum(row["job_type"] == "upper_bound" for row in rows) == 8
    assert sum(row["job_type"] == "ablation" for row in rows) == 48
    assert 2 * sum(row["job_type"] == "ablation" for row in rows) == 96


def test_prompt_quota_and_presence_balancing():
    names = ("a", "b", "c")
    assert _quota(32, names) == {"a": 11, "b": 11, "c": 10}
    records = [
        {"image_id": f"p{i}", "present_classes": ["a"]} for i in range(10)
    ] + [{"image_id": f"n{i}", "present_classes": []} for i in range(10)]
    import random
    chosen = _balanced_prompt_sample(records, "a", 8, random.Random(1), True)
    assert len(chosen) == 8
    assert sum("a" in row["present_classes"] for row in chosen) == 4


def test_cnn_uses_dense_representation_and_starts_at_reference_mean():
    cfg = load_config()
    device = torch.device("cpu")
    alpha, beta = reference_beta(cfg, "BrainMRI", device)
    policy = CNNBetaPolicy(embedding_dim=8, hidden=4, action_dim=4, beta_floor=0.5, beta_max=100)
    policy.initialize(alpha, beta)
    local = torch.randn(3, 4, 8)
    text = torch.randn(3, 8)
    predicted_alpha, predicted_beta = policy.parameters_ab(local, text, (2, 2))
    action = map_actions(predicted_alpha / (predicted_alpha + predicted_beta), cfg)
    expected = torch.tensor([1.4, 0.6, 0.5, 0.0]).expand_as(action)
    # Boundary actions are represented by the configured 0.02 Beta-mean clamp.
    assert torch.allclose(action[:, :3], expected[:, :3], atol=1e-5)
    assert torch.allclose(action[:, 3], torch.full((3,), 0.08), atol=1e-5)


def test_exp0_reward_has_only_positive_dice_and_absent_fp_cases():
    base = dict(
        dataset="dummy", image_id="x", class_name="c", class_idx=0,
        local=torch.zeros(4, 2), text_delta=torch.zeros(2),
        hat=torch.tensor([2.0, -1.0, -1.0, -1.0]), seed_idx=0,
        base_affinity=torch.ones(4),
        coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        grid_shape=(2, 2),
    )
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    positive = Pair(present=True, gt_mask=torch.tensor([[True, False], [False, False]]), **base)
    absent = Pair(present=False, gt_mask=torch.zeros((2, 2), dtype=torch.bool), **base)
    assert torch.allclose(hard_reward(positive, action), torch.ones(1))
    assert torch.allclose(hard_reward(absent, action), torch.tensor([0.75]))
