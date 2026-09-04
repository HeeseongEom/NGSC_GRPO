#!/usr/bin/env python3
"""Cache one class-agnostic prompt-ensemble localization disagreement scalar."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from common import ROOT, _cache_index, exp2_root, load_exp2_config, read_csv, write_csv
from ngsc_grpo.model_adapter import DenseBiomedCLIP
from ngsc_grpo.registry import get_spec


@torch.inference_mode()
def image_features_batch(extractor: DenseBiomedCLIP, images: list[Image.Image]):
    pixels = extractor.processor(images=[image.convert("RGB") for image in images], return_tensors="pt")[
        "pixel_values"
    ].to(extractor.device)
    vision = extractor.model.vision_model
    hidden = vision.embeddings(pixels)
    for layer in vision.encoder.layers[:-1]:
        hidden = layer(hidden, None, False, tsa=False)[0]
    count = hidden.shape[1] - 1
    side = int(round(count ** 0.5))
    if side * side != count:
        raise ValueError(count)
    dense = extractor._dense_last_block(hidden, (side, side))
    local = extractor.model.visual_projection(dense[:, 1:])
    return F.normalize(local.float(), dim=-1)


@torch.inference_mode()
def template_features(extractor: DenseBiomedCLIP, dataset_name: str):
    spec = get_spec(dataset_name)
    normal_prompts = extractor.templates[spec.normal_class][:50]
    if len(normal_prompts) < 50:
        raise ValueError(f"{spec.normal_class} has fewer than 50 templates")
    normal = extractor._encode_prompts(normal_prompts)
    output = {}
    for class_name in spec.foreground_classes:
        prompts = extractor.templates[class_name][:50]
        if len(prompts) < 50:
            raise ValueError(f"{class_name} has fewer than 50 templates")
        output[class_name] = (extractor._encode_prompts(prompts), normal)
    return output


@torch.inference_mode()
def js_values(local: torch.Tensor, text_pairs, temperature: float):
    output = {}
    for class_name, (foreground, normal) in text_pairs.items():
        contrast = torch.einsum("bnd,md->bnm", local, foreground) - torch.einsum(
            "bnd,md->bnm", local, normal
        )
        contrast = contrast.permute(0, 2, 1)
        contrast = (contrast - contrast.mean(-1, keepdim=True)) / (
            contrast.std(-1, keepdim=True, unbiased=False) + 1e-6
        )
        distributions = torch.softmax(contrast / float(temperature), dim=-1)
        mean = distributions.mean(1, keepdim=True)
        js = (distributions * (torch.log(distributions + 1e-12) - torch.log(mean + 1e-12))).sum(-1).mean(-1)
        output[class_name] = js
    return output


def run(cfg, method: str, device: str, initial_batch: int, force: bool):
    output = exp2_root(cfg) / "prompt_disagreement" / method / "values.csv"
    existing = {} if force or not output.is_file() else {row["key"]: row for row in read_csv(output)}
    rows_by_dataset = defaultdict(list)
    for split in ("source", "internal", "external"):
        for row in read_csv(_cache_index(cfg, method, split)):
            row = dict(row)
            row["split"] = split
            rows_by_dataset[row["dataset"]].append(row)
    extractor = DenseBiomedCLIP(cfg={
        **cfg,
        "runtime": {"device": device},
        "paths": {
            "checkpoint_dir": "checkpoints/biomedclip_ngsc",
            "prompt_template": "assets/prompts/biomedcoop_templates.py",
        },
        "ngsc_core": {
            "naclip_gaussian_std": 5.0,
            "cdam_temperature": 0.8,
            "cdam_softmax_temperature": 0.8,
        },
        "dense_methods": cfg["methods"],
        "_project_root": str(ROOT),
    }, method=method, device=device)
    data_root = ROOT / "data"
    batch_size = int(initial_batch)
    for dataset, entries in rows_by_dataset.items():
        text_pairs = template_features(extractor, dataset)
        pending = [
            row for row in entries
            if any("|".join((dataset, row["image_id"], name)) not in existing for name in text_pairs)
        ]
        cursor = 0
        while cursor < len(pending):
            current = min(batch_size, len(pending) - cursor)
            subset = pending[cursor:cursor + current]
            images = [Image.open(data_root / row["image_relpath"]).convert("RGB") for row in subset]
            try:
                local = image_features_batch(extractor, images)
                values = js_values(local, text_pairs, temperature=1.0)
            except torch.cuda.OutOfMemoryError:
                del images
                torch.cuda.empty_cache()
                gc.collect()
                if current <= 8:
                    raise
                batch_size = max(8, current // 2)
                print(json.dumps({"method": method, "oom_reduced_batch": batch_size}), flush=True)
                continue
            for image_idx, row in enumerate(subset):
                for class_name, tensor in values.items():
                    key = "|".join((dataset, row["image_id"], class_name))
                    existing[key] = {
                        "key": key, "method": method, "split": row["split"], "dataset": dataset,
                        "image_id": row["image_id"], "class_name": class_name,
                        "prompt_js": float(tensor[image_idx].cpu()),
                    }
            cursor += current
            allocated = torch.cuda.max_memory_allocated(extractor.device) / (1024 ** 3)
            print(json.dumps({"method": method, "dataset": dataset, "done": cursor, "total": len(pending), "batch": current, "max_gpu_GiB": round(allocated, 2)}), flush=True)
            del local, values, images
        write_csv(output, sorted(existing.values(), key=lambda row: row["key"]))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    print(run(cfg, args.method, args.device, args.batch_size, args.force))


if __name__ == "__main__":
    main()
