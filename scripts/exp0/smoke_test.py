#!/usr/bin/env python3
"""One real image through pre-SCM features, both policies, and the two-case reward."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import (
    CNNBetaPolicy,
    GlobalBetaPolicy,
    Pair,
    data_root,
    hard_reward,
    load_config,
    map_actions,
    policy_distribution,
    records_for_dataset,
    reference_beta,
)
from ngsc_grpo.core import normalized_coords, per_class_normalize
from ngsc_grpo.model_adapter import DenseBiomedCLIP
from ngsc_grpo.registry import get_spec, load_class_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="BrainMRI")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    device = torch.device(args.device)
    dataset = args.dataset
    spec = get_spec(dataset)
    record = records_for_dataset(cfg, dataset)[0]
    extractor = DenseBiomedCLIP({
        **cfg,
        "dense_methods": ["MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP"],
        "runtime": {"device": args.device},
        "_project_root": str(data_root(cfg).parent),
    }, cfg["method"], device=args.device)
    image = Image.open(data_root(cfg) / record["image_relpath"]).convert("RGB")
    with torch.inference_mode():
        class_text, cdam_text = extractor.text_features(spec)
        cls_original, local_batch, grid_shape = extractor.image_features(image)
        local = local_batch[0]
        scores = local @ class_text.T
        raw = scores[:, :-1].T - scores[:, -1].unsqueeze(0)
        hat = per_class_normalize(raw)
        seeds = hat.argmax(-1)
        cdam_features = torch.cat((cls_original, local), dim=0)
        cdam = extractor._cdam_matrix(
            cdam_features, cdam_text, cfg["ngsc_core"]["cdam_temperature"],
            cfg["ngsc_core"]["cdam_softmax_temperature"],
        )
        affinity = torch.stack([cdam[int(seed) + 1] for seed in seeds])
        text_delta = F.normalize(class_text[:-1] - class_text[-1][None], dim=-1)
    masks = load_class_masks(
        data_root(cfg), record,
        size=(int(cfg["cache"]["reward_resolution"]), int(cfg["cache"]["reward_resolution"])),
    )
    class_idx = 0
    gt = torch.from_numpy(np.asarray(masks[spec.foreground_classes[class_idx]])).bool()
    pair = Pair(
        dataset, record["image_id"], spec.foreground_classes[class_idx], class_idx, bool(gt.any()),
        local.cpu(), text_delta[class_idx].cpu(), hat[class_idx].cpu(), int(seeds[class_idx]),
        affinity[class_idx].cpu(), normalized_coords(*grid_shape).cpu(), gt, grid_shape,
    )
    alpha, beta = reference_beta(cfg, dataset, device)
    global_policy = GlobalBetaPolicy(4, cfg["policy"]["beta_floor"], cfg["policy"]["beta_max"]).to(device)
    global_policy.initialize(alpha, beta)
    cnn = CNNBetaPolicy(
        local.shape[-1], cfg["policy"]["cnn_hidden_channels"], 4,
        cfg["policy"]["beta_floor"], cfg["policy"]["beta_max"],
    ).to(device)
    cnn.initialize(alpha, beta)
    ga, gb = global_policy.parameters_ab(1)
    ca, cb = cnn.parameters_ab(local[None], text_delta[class_idx][None], grid_shape)
    global_action = map_actions(ga / (ga + gb), cfg)
    cnn_action = map_actions(ca / (ca + cb), cfg)
    reward_global = hard_reward(pair, global_action)
    reward_cnn = hard_reward(pair, cnn_action)
    train_step_finite = True
    for kind, policy in (("global", global_policy), ("cnn", cnn)):
        batch = [pair, pair]
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        with torch.no_grad():
            old = policy_distribution(policy, kind, batch, device)
            frozen = torch.distributions.Beta(
                old.concentration1.detach().clone(), old.concentration0.detach().clone()
            )
            normalized = frozen.sample((4,)).permute(1, 0, 2).contiguous()
            old_log_prob = frozen.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
            physical = map_actions(normalized, cfg)
            rewards = torch.stack([hard_reward(value, physical[index]) for index, value in enumerate(batch)])
            spread = rewards.std(1, keepdim=True, unbiased=False)
            advantage = torch.where(
                spread > 1e-6, (rewards - rewards.mean(1, keepdim=True)) / (spread + 1e-6),
                torch.zeros_like(rewards),
            )
        current = policy_distribution(policy, kind, batch, device)
        log_prob = current.log_prob(normalized.permute(1, 0, 2)).sum(-1).permute(1, 0)
        loss = -(torch.exp(log_prob - old_log_prob) * advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        train_step_finite = train_step_finite and bool(torch.isfinite(loss) and torch.isfinite(gradient))
    result = {
        "ok": bool(torch.isfinite(reward_global).all() and torch.isfinite(reward_cnn).all()
                   and train_step_finite),
        "dataset": dataset,
        "method": cfg["method"],
        "grid_shape": list(grid_shape),
        "embedding_dim": int(local.shape[-1]),
        "classes": len(spec.foreground_classes),
        "cnn_parameters": sum(parameter.numel() for parameter in cnn.parameters()),
        "global_action": global_action[0].detach().cpu().tolist(),
        "cnn_action": cnn_action[0].detach().cpu().tolist(),
        "global_reward": float(reward_global[0]),
        "cnn_reward": float(reward_cnn[0]),
        "one_grpo_update_finite": train_step_finite,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
