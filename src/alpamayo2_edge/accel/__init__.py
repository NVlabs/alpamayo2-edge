# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in inference acceleration for Alpamayo 2 Edge.

``enable_acceleration`` installs ``torch.compile`` plus CUDA graphs on one loaded
model: the vision tower, the prompt prefill, CoT decoding with a graphed sampler
and the whole flow-matching loop. ``use_linear_patch_embedding`` computes the
patch embedding as a GEMM. Nothing here changes the default inference path.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch

from alpamayo2_edge.accel.decode import DecodeAccelerator, FlashKVCache
from alpamayo2_edge.accel.expert import ExpertAccelerator
from alpamayo2_edge.accel.graphs import AccelStats
from alpamayo2_edge.accel.kernels import ATTENTION_CHOICES, PARTS, resolve_kernels
from alpamayo2_edge.accel.patch_embed import use_linear_patch_embedding
from alpamayo2_edge.accel.vision import VisionAccelerator

MODES = ("compile", "tuned")
# Inductor mode of the decode step; every other part uses the default mode.
_DECODE_MODES = {"compile": "default", "tuned": "max-autotune-no-cudagraphs"}


@dataclass(frozen=True)
class AccelConfig:
    """Acceleration settings. The defaults are the validated configuration.

    ``mode="tuned"`` compiles the decode step with inductor's
    ``max-autotune-no-cudagraphs`` mode, and ``"compile"`` uses the default mode
    everywhere. ``attention`` selects the vision and prefill kernel: ``"auto"`` uses
    FlashAttention 3 on Hopper when it is installed, and PyTorch SDPA otherwise.
    Decode and expert attention always use FlashAttention 2 ``flash_attn_with_kvcache``.
    Shapes are bucketed to ``kv_bucket`` tokens. ``compile=False`` and
    ``cuda_graphs=False`` run the same code eagerly, for debugging and CPU tests.
    """

    mode: str = "tuned"
    attention: str = "auto"
    parts: tuple[str, ...] = PARTS
    kv_bucket: int = 256
    warmup: int = 2
    compile: bool = True
    cuda_graphs: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.attention not in ATTENTION_CHOICES:
            raise ValueError(f"attention must be one of {ATTENTION_CHOICES}")
        unknown = set(self.parts) - set(PARTS)
        if unknown:
            raise ValueError(f"Unknown parts {sorted(unknown)}; choose from {PARTS}")
        if self.kv_bucket < 1 or self.warmup < 0:
            raise ValueError("kv_bucket must be positive and warmup nonnegative")


class AccelHandle:
    """Installed accelerators of one model: their settings, graph builds and fallbacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: AccelConfig,
        kernels: dict[str, str | None],
        skipped: dict[str, str],
        stats: AccelStats,
        accelerators: list[Any],
    ) -> None:
        self.model, self.config, self.kernels = model, config, kernels
        self.skipped, self.stats, self.accelerators = skipped, stats, accelerators
        self.parts = tuple(part for part in PARTS if part in config.parts and part not in skipped)

    def describe(self) -> dict[str, Any]:
        """JSON-ready settings, resolved kernels, graph builds and fallback counts."""
        return {
            "config": asdict(self.config),
            "parts": list(self.parts),
            "skipped": dict(self.skipped),
            "kernels": {part: self.kernels[part] for part in self.parts},
            **self.stats.as_dict(),
        }

    def warmup(self, inputs: dict[str, Any], **sample_kwargs: Any) -> None:
        """Compile and capture the graphs for ``inputs`` without changing RNG state."""
        device = self.model.device
        with (
            torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
            torch.inference_mode(),
            torch.autocast(device.type, dtype=self.model.dtype, enabled=device.type == "cuda"),
        ):
            self.model.sample_trajectories_from_data(inputs, **sample_kwargs)

    def remove(self) -> None:
        """Restore the original methods (wrappers installed later must be removed first)."""
        for accelerator in reversed(self.accelerators):
            accelerator.remove()
        self.accelerators = []
        self.model._alpamayo2_edge_accel = None


def enable_acceleration(model: torch.nn.Module, config: AccelConfig | None = None) -> AccelHandle:
    """Install the selected accelerators on ``model`` (an ``Alpamayo2Edge`` instance).

    The first call for each new shape compiles and captures graphs inline (tens of
    seconds); later calls replay them. Inputs outside the supported envelope run
    the original eager code, counted in ``handle.describe()["fallbacks"]``.
    """
    config = config or AccelConfig()
    if getattr(model, "_alpamayo2_edge_accel", None) is not None:
        raise RuntimeError("Acceleration is already enabled on this model")
    if not model.config.enable_expert:
        raise ValueError("Acceleration requires the trajectory expert")
    device = model.device
    if config.cuda_graphs and device.type != "cuda":
        raise ValueError("CUDA graphs require a model on a CUDA device")
    if device.type == "cuda" and model.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FlashAttention kernels require a BF16 or FP16 model")
    kernels = resolve_kernels(config.attention, device)
    requested = set(config.parts)
    skipped: dict[str, str] = {}
    for part in ("decode", "expert"):
        if part in requested and kernels[part] is None:
            skipped[part] = "requires flash_attn (FlashAttention 2)"
    # Prefill writes, and the expert reads, the decode part's flash KV cache.
    has_decode = "decode" in requested and "decode" not in skipped
    for part in ("prefill", "expert"):
        if part in requested and part not in skipped and not has_decode:
            skipped[part] = "requires the decode part"
    parts = [part for part in PARTS if part in requested and part not in skipped]
    dynamo_config = torch._dynamo.config
    # One compiled function serves every shape of its part; allow a recompile per shape.
    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
    stats = AccelStats()
    common = dict(
        stats=stats,
        compile=config.compile,
        cuda_graphs=config.cuda_graphs,
        warmup=config.warmup,
    )
    accelerators: list[Any] = []
    if "decode" in parts:
        accelerators.append(
            DecodeAccelerator(
                model,
                **common,
                kv_bucket=config.kv_bucket,
                decode_mode=_DECODE_MODES[config.mode],
                mode="default",
                graph_prefill="prefill" in parts,
                prefill_fa3=kernels["prefill"] == "fa3",
            )
        )
    if "expert" in parts:
        accelerators.append(ExpertAccelerator(model, **common, mode="default"))
    if "vision" in parts:
        accelerators.append(
            VisionAccelerator(
                model.vlm.model.visual, **common, mode="default", fa3=kernels["vision"] == "fa3"
            )
        )
    handle = AccelHandle(model, config, kernels, skipped, stats, accelerators)
    model._alpamayo2_edge_accel = handle
    return handle


__all__ = [
    "AccelConfig",
    "AccelHandle",
    "FlashKVCache",
    "enable_acceleration",
    "resolve_kernels",
    "use_linear_patch_embedding",
]
