"""Utilities to synthesize lightweight LoRA adapters for benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


@dataclass(frozen=True)
class DummyLoRAResult:
    path: Path
    adapter_id: str
    rank: int


def _sanitize_name(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model_id)


def _infer_ffn_dim(cfg) -> int:
    if hasattr(cfg, "intermediate_size") and cfg.intermediate_size is not None:
        return int(cfg.intermediate_size)
    if hasattr(cfg, "ffn_hidden_size") and cfg.ffn_hidden_size is not None:
        return int(cfg.ffn_hidden_size)
    raise ValueError("Model config does not define intermediate/ffn hidden size.")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_dummy_lora(
    model_id: str,
    *,
    rank: int,
    root: Path,
    seed: int = 0,
    force: bool = False,
) -> DummyLoRAResult:
    """Create a dummy LoRA adapter directory for a given model.

    The generated adapter only targets the projection layers that are common
    across decoder-only architectures (``o_proj`` and ``down_proj``), which is
    sufficient for exercising the LoRA runtime paths during benchmarks while
    avoiding large disk footprints.
    """
    safe_model = _sanitize_name(model_id)
    adapter_dir = Path(root) / safe_model / f"rank_{rank}"
    if adapter_dir.exists() and not force:
        adapter_id = f"{safe_model}_r{rank}"
        return DummyLoRAResult(path=adapter_dir, adapter_id=adapter_id, rank=rank)

    cfg = AutoConfig.from_pretrained(model_id)
    hidden_size = int(cfg.hidden_size)
    intermediate_size = _infer_ffn_dim(cfg)
    num_layers = int(cfg.num_hidden_layers)

    generator = torch.Generator().manual_seed(seed)
    weights: dict[str, torch.Tensor] = {}
    module_specs: Tuple[Tuple[str, int, int], ...] = (
        ("self_attn.o_proj", hidden_size, hidden_size),
        ("mlp.down_proj", intermediate_size, hidden_size),
    )

    _ensure_dir(adapter_dir)

    for layer_idx in range(num_layers):
        for module_path, in_features, out_features in module_specs:
            prefix = (
                f"base_model.model.model.layers."
                f"{layer_idx}.{module_path}"
            )
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            weights[a_key] = torch.randn(
                rank,
                in_features,
                dtype=torch.float16,
                generator=generator,
            )
            weights[b_key] = torch.randn(
                out_features,
                rank,
                dtype=torch.float16,
                generator=generator,
            )

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))

    adapter_config = {
        "base_model_name_or_path": model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "layers_pattern": "model.layers.",
        "lora_alpha": max(16, rank * 2),
        "lora_dropout": 0.0,
        "modules_to_save": [],
        "peft_type": "LORA",
        "r": rank,
        "target_modules": ["o_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
    }

    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as fp:
        json.dump(adapter_config, fp, indent=2)

    (adapter_dir / "README.txt").write_text(
        "This directory contains randomly initialised LoRA weights for "
        "benchmarking only. Do not use for training or deployment.\n",
        encoding="utf-8",
    )

    adapter_id = f"{safe_model}_r{rank}"
    return DummyLoRAResult(path=adapter_dir, adapter_id=adapter_id, rank=rank)
