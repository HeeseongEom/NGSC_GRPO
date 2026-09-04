#!/usr/bin/env python3
"""Deterministic unit checks for the EXP0_1 reward branches and components."""

from __future__ import annotations

import numpy as np
import torch

from common import Pair, hard_reward, load_config


def pair_from_masks(prediction: torch.Tensor, target: torch.Tensor, present: bool) -> Pair:
    height, width = prediction.shape
    return Pair(
        dataset="synthetic",
        image_id="synthetic",
        class_name="lesion",
        class_idx=0,
        present=present,
        local=torch.zeros(height * width, 1),
        text_delta=torch.zeros(1),
        hat=prediction.float().reshape(-1),
        seed_idx=0,
        base_affinity=torch.ones(height * width),
        coords=torch.zeros(height * width, 2),
        gt_mask=target.bool(),
        grid_shape=(height, width),
    )


def main() -> None:
    cfg = load_config()
    reward_cfg = cfg["reward"]
    actions = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    empty = torch.zeros(8, 8, dtype=torch.bool)
    square = empty.clone()
    square[2:5, 2:5] = True

    perfect = float(hard_reward(pair_from_masks(square, square, True), actions, reward_cfg)[0])
    missed = float(hard_reward(pair_from_masks(empty, square, True), actions, reward_cfg)[0])
    absent_empty = float(hard_reward(pair_from_masks(empty, empty, False), actions, reward_cfg)[0])
    one_pixel = empty.clone()
    one_pixel[3, 3] = True
    absent_fp = float(hard_reward(pair_from_masks(one_pixel, empty, False), actions, reward_cfg)[0])
    expected_fp = (1.0 - 1.0 / 64.0) * (1.0 - float(reward_cfg["absent_margin"]))

    values = np.asarray([perfect, missed, absent_empty, absent_fp])
    assert np.isfinite(values).all(), values
    assert abs(perfect - 1.0) < 1e-6, perfect
    assert abs(missed) < 1e-6, missed
    assert abs(absent_empty - 1.0) < 1e-6, absent_empty
    assert abs(absent_fp - expected_fp) < 1e-6, (absent_fp, expected_fp)
    print({
        "perfect_present": perfect,
        "missed_present": missed,
        "empty_absent": absent_empty,
        "one_pixel_fp_absent": absent_fp,
        "expected_one_pixel_fp_absent": expected_fp,
    })


if __name__ == "__main__":
    main()
