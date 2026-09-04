from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .config import project_path, require_method
from .core import normalized_coords, per_class_normalize
from .registry import DatasetSpec


def _load_templates(path: Path) -> Mapping[str, list[str]]:
    spec = importlib.util.spec_from_file_location("ngsc_biomedcoop_templates", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import prompt templates from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = getattr(module, "BIOMEDCOOP_TEMPLATES")
    return templates


class DenseBiomedCLIP:
    """Frozen BiomedCLIP with explicit, runtime-selectable dense attention adapters.

    The legacy repository exposed four method names in shell scripts while its copied
    model file hard-coded one TSA branch. This adapter executes the first 11 blocks
    normally, then applies the requested final-block rule without changing weights.
    """

    def __init__(self, cfg, method: str, device: str | None = None):
        require_method(method)
        self.cfg = cfg
        self.method = method
        self.device = torch.device(device or cfg["runtime"]["device"])
        checkpoint_dir = project_path(cfg, cfg["paths"]["checkpoint_dir"])
        if not (checkpoint_dir / "pytorch_model.bin").is_file():
            raise FileNotFoundError(f"Missing copied fine-tuned checkpoint: {checkpoint_dir}")
        self.model = AutoModel.from_pretrained(
            checkpoint_dir, trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_dir, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True, local_files_only=True
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.templates = _load_templates(project_path(cfg, cfg["paths"]["prompt_template"]))
        self._text_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _encode_prompts(self, prompts, batch_size: int = 64) -> torch.Tensor:
        features = []
        for start in range(0, len(prompts), batch_size):
            encoded = self.tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            feature = self.model.get_text_features(**encoded)
            features.append(F.normalize(feature.float(), dim=-1))
        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def text_features(self, dataset: DatasetSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        if dataset.name in self._text_cache:
            return self._text_cache[dataset.name]
        all_names = list(dataset.foreground_classes) + [dataset.normal_class]
        class_features = []
        for name in all_names:
            prompts = self.templates[name]
            if len(prompts) < 50:
                raise ValueError(f"Prompt class {name!r} has fewer than 50 templates")
            features = self._encode_prompts(prompts[:50])
            class_features.append(F.normalize(features.mean(dim=0, keepdim=True), dim=-1))
        ensemble = torch.cat(class_features, dim=0)

        normal_prompts = self.templates[dataset.normal_prompt_key][:50]
        abnormal_prompts = self.templates[dataset.abnormal_prompt_key][:50]
        cdam_prompts = []
        for normal, abnormal in zip(normal_prompts, abnormal_prompts):
            cdam_prompts.extend((normal, abnormal))
        cdam = self._encode_prompts(cdam_prompts)
        self._text_cache[dataset.name] = (ensemble, cdam)
        return ensemble, cdam

    @staticmethod
    def _project_qkv(attn, hidden: torch.Tensor):
        q = attn.transpose_for_scores(attn.q_proj(hidden))
        k = attn.transpose_for_scores(attn.k_proj(hidden))
        v = attn.transpose_for_scores(attn.v_proj(hidden))
        return q, k, v

    @staticmethod
    def _merge_heads(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 1, 3).contiguous().reshape(value.shape[0], value.shape[2], -1)

    @staticmethod
    def _gaussian_addition(grid_shape, std, device, dtype):
        height, width = grid_shape
        ys = torch.arange(height, device=device, dtype=torch.float32)
        xs = torch.arange(width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        dist2 = ((coords[:, None] - coords[None]) ** 2).sum(dim=-1)
        patch_addition = torch.exp(-dist2 / (2.0 * float(std) ** 2))
        out = torch.zeros((height * width + 1, height * width + 1), device=device, dtype=torch.float32)
        out[1:, 1:] = patch_addition
        return out.to(dtype=dtype)

    def _dense_last_block(self, hidden: torch.Tensor, grid_shape) -> torch.Tensor:
        layer = self.model.vision_model.encoder.layers[-1]
        normalized = layer.layer_norm1(hidden)
        attn = layer.self_attn
        q, k, v = self._project_qkv(attn, normalized)
        scale = 1.0 / math.sqrt(float(attn.head_dim))

        if self.method == "MaskCLIP":
            # Extract Free Dense Labels: retain the final value/output projections,
            # without the globalizing QK attention, residual, or FFN.
            return attn.out_proj(self._merge_heads(v))

        if self.method == "SCLIP":
            q_attn = torch.matmul(q, q.transpose(-1, -2)) * scale
            k_attn = torch.matmul(k, k.transpose(-1, -2)) * scale
            weights = torch.softmax(q_attn, dim=-1) + torch.softmax(k_attn, dim=-1)
            attention_output = attn.out_proj(self._merge_heads(torch.matmul(weights, v)))
            output = hidden + attention_output
            return output + layer.mlp(layer.layer_norm2(output))

        if self.method == "ClearCLIP":
            # Final-block self-self attention only; residual and FFN are removed.
            q_attn = torch.matmul(q, q.transpose(-1, -2)) * scale
            weights = torch.softmax(q_attn, dim=-1)
            return attn.out_proj(self._merge_heads(torch.matmul(weights, v)))

        if self.method == "NACLIP":
            k_attn = torch.matmul(k, k.transpose(-1, -2)) * scale
            gaussian = self._gaussian_addition(
                grid_shape,
                self.cfg["ngsc_core"]["naclip_gaussian_std"],
                k_attn.device,
                k_attn.dtype,
            )
            weights = torch.softmax((k_attn + gaussian[None, None]) * scale, dim=-1)
            return attn.out_proj(self._merge_heads(torch.matmul(weights, v)))
        raise AssertionError(self.method)

    @torch.inference_mode()
    def image_features_batch(
        self, images: list[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Extract frozen CLS and dense features for a same-resolution model batch."""
        if not images:
            raise ValueError("images must not be empty")
        pixels = self.processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )["pixel_values"].to(self.device)
        vision = self.model.vision_model
        hidden = vision.embeddings(pixels)
        for layer in vision.encoder.layers[:-1]:
            hidden = layer(hidden, None, False, tsa=False)[0]

        standard_hidden = vision.encoder.layers[-1](hidden, None, False, tsa=False)[0]
        cls_original = self.model.visual_projection(vision.post_layernorm(standard_hidden[:, 0]))

        num_patches = hidden.shape[1] - 1
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(f"Only square patch grids are supported, got {num_patches} patches")
        dense_hidden = self._dense_last_block(hidden, (side, side))
        local = self.model.visual_projection(dense_hidden[:, 1:])
        return F.normalize(cls_original.float(), dim=-1), F.normalize(local.float(), dim=-1), (side, side)

    @torch.inference_mode()
    def image_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        return self.image_features_batch([image])

    @staticmethod
    def _cdam_matrix(features: torch.Tensor, cdam_text: torch.Tensor, temperature: float, softmax_temperature: float):
        distributions = torch.softmax((features @ cdam_text.T) / float(temperature), dim=-1)
        p = distributions[:, None, :]
        q = distributions[None, :, :]
        midpoint = 0.5 * (p + q)
        js = 0.5 * (
            (p * (torch.log(p + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=-1)
            + (q * (torch.log(q + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=-1)
        )
        similarity = 1.0 - js
        similarity = similarity[:, 1:]
        minimum = similarity.min(dim=-1, keepdim=True).values
        maximum = similarity.max(dim=-1, keepdim=True).values
        similarity = (similarity - minimum) / (maximum - minimum + 1e-8)
        similarity = torch.softmax(similarity / float(softmax_temperature), dim=-1)
        minimum = similarity.min(dim=-1, keepdim=True).values
        maximum = similarity.max(dim=-1, keepdim=True).values
        return (similarity - minimum) / (maximum - minimum + 1e-8)

    @torch.inference_mode()
    def frozen_ngsc_quantities(self, image: Image.Image, dataset: DatasetSpec) -> dict:
        class_text, cdam_text = self.text_features(dataset)
        cls_original, local, grid_shape = self.image_features(image)
        scores = local[0] @ class_text.T
        raw = scores[:, :-1].T - scores[:, -1].unsqueeze(0)
        hat = per_class_normalize(raw)
        seeds = hat.argmax(dim=-1)

        cdam_features = torch.cat((cls_original, local[0]), dim=0)
        cdam = self._cdam_matrix(
            cdam_features,
            cdam_text,
            self.cfg["ngsc_core"]["cdam_temperature"],
            self.cfg["ngsc_core"]["cdam_softmax_temperature"],
        )
        base_affinity = torch.stack([cdam[int(seed) + 1] for seed in seeds], dim=0)
        if not bool(torch.isfinite(base_affinity).all()):
            raise FloatingPointError("Non-finite CDAM affinity")
        if float(base_affinity.min()) < -1e-5 or float(base_affinity.max()) > 1.00001:
            raise AssertionError("Base CDAM affinity is outside [0,1]")
        coords = normalized_coords(*grid_shape, device=raw.device, dtype=raw.dtype)
        return {
            "raw": raw,
            "hat": hat,
            "seed_idx": seeds,
            "base_affinity": base_affinity,
            "coords": coords,
            "grid_shape": grid_shape,
        }
