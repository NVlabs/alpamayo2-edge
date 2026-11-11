# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention kernels used inside the accelerated graphs, and their selection.

Decode and expert steps attend a static flash-layout KV cache through
``flash_attn_with_kvcache`` from FlashAttention 2 or, when only FlashAttention 4
(``flash_attn.cute``, e.g. on Blackwell/Thor) is installed, through FA4's
``flash_attn_varlen_func`` after an in-place cache write. The prompt prefill uses PyTorch
SDPA or, on Hopper with FlashAttention 3 installed, FA3. Both kernels are custom
ops, so ``torch.compile`` keeps each as one opaque kernel. CPU tensors use a
PyTorch reference implementation, which keeps the control flow testable without
a GPU; it is not used for CUDA tensors.
"""

import functools
import importlib

import torch
import torch.nn.functional as F

PARTS = ("prefill", "decode", "expert")
ATTENTION_CHOICES = ("auto", "fa3", "sdpa")


@functools.cache
def _flash_function(module_name: str, function_name: str):
    return getattr(importlib.import_module(module_name), function_name)


def _kvcache_attention_reference(q, k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal):
    out = torch.empty_like(q)
    new_tokens = k.shape[1]
    for row, start in enumerate(cache_seqlens.tolist()):
        end = start + new_tokens
        k_cache[row, start:end] = k[row]
        v_cache[row, start:end] = v[row]
        mask = None
        if causal:
            # FlashAttention aligns the causal mask to the last cache column.
            mask = torch.ones(q.shape[1], end, dtype=torch.bool, device=q.device)
            mask = mask.tril(end - q.shape[1])
        out[row] = F.scaled_dot_product_attention(
            q[row].transpose(0, 1),
            k_cache[row, :end].transpose(0, 1),
            v_cache[row, :end].transpose(0, 1),
            attn_mask=mask,
            scale=softmax_scale,
            enable_gqa=True,
        ).transpose(0, 1)
    return out


def _kvcache_attention_fa4(q, k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal):
    # FA4 has no fused cache append: write the new tokens with graph-safe indexing first.
    new_tokens = k.shape[1]
    rows = torch.arange(q.shape[0], device=q.device)[:, None]
    columns = cache_seqlens.long()[:, None] + torch.arange(new_tokens, device=q.device)
    k_cache[rows, columns] = k
    v_cache[rows, columns] = v
    out = _flash_function("flash_attn.cute", "flash_attn_varlen_func")(
        q,
        k_cache,
        v_cache,
        seqused_k=cache_seqlens + new_tokens,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.library.custom_op("alpamayo2_edge::kvcache_attention", mutates_args=("k_cache", "v_cache"))
def kvcache_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Write ``k``/``v`` at ``cache_seqlens`` in place and attend the valid cache prefix.

    Layouts follow FlashAttention: ``q`` is ``[B, Tq, Hq, D]``, the caches are
    ``[B, max_len, Hkv, D]``, ``k``/``v`` are ``[B, Tk, Hkv, D]`` and
    ``cache_seqlens`` is int32 ``[B]``. Row ``b`` attends cache columns
    ``[0, cache_seqlens[b] + Tk)``; ``causal`` aligns the mask to the last column.
    """
    if not q.is_cuda:
        return _kvcache_attention_reference(
            q, k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal
        )
    if _kvcache_kernel() == "fa4":
        out = _kvcache_attention_fa4(
            q, k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal
        )
        return out.contiguous()
    out = _flash_function("flash_attn", "flash_attn_with_kvcache")(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out.contiguous()


@kvcache_attention.register_fake
def _kvcache_attention_fake(q, k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal):
    del k_cache, v_cache, k, v, cache_seqlens, softmax_scale, causal
    return torch.empty_like(q, memory_format=torch.contiguous_format)


@torch.library.custom_op("alpamayo2_edge::fa3_attention", mutates_args=())
def fa3_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> torch.Tensor:
    """FlashAttention 3 ``flash_attn_func`` on ``[B, S, H, D]`` tensors (GQA by head count)."""
    if not q.is_cuda:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=True,
        ).transpose(1, 2)
    out = _flash_function("flash_attn_interface", "flash_attn_func")(
        q, k, v, softmax_scale=softmax_scale, causal=causal
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.contiguous()


@fa3_attention.register_fake
def _fa3_attention_fake(q, k, v, softmax_scale, causal):
    del k, v, softmax_scale, causal
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a broken CUDA extension is as unusable as a missing one
        return False
    return True


def _select_kvcache_kernel() -> str | None:
    # FA4 wheels install a ``flash_attn`` namespace holding only ``flash_attn.cute``, so
    # FA2 is detected by its interface module rather than by ``flash_attn`` itself.
    if _importable("flash_attn.flash_attn_interface"):
        return "fa2"
    if _importable("flash_attn.cute"):
        return "fa4"
    return None


@functools.cache
def _kvcache_kernel() -> str | None:
    return _select_kvcache_kernel()


def resolve_kernels(attention: str, device: torch.device | str) -> dict[str, str | None]:
    """Choose the attention kernel of each accelerated part.

    ``prefill`` uses ``"fa3"`` or ``"sdpa"``. ``"auto"`` picks FA3 only on
    Hopper (compute capability 9.0) with ``flash_attn_interface`` importable.
    ``decode`` and ``expert`` use ``"fa2"`` when FlashAttention 2 is importable, else
    ``"fa4"`` when FlashAttention 4 (``flash_attn.cute``) is, otherwise ``None``
    (those parts stay eager). FA2 is kept for one-token decode on Hopper too,
    where it is faster than FA3. CPU devices use the PyTorch ``"reference"`` kernels.
    """
    if attention not in ATTENTION_CHOICES:
        raise ValueError(f"attention must be one of {ATTENTION_CHOICES}, got {attention!r}")
    device = torch.device(device)
    if device.type != "cuda":
        if attention == "fa3":
            raise ValueError("FlashAttention 3 requires a CUDA device")
        return {"prefill": "sdpa", "decode": "reference", "expert": "reference"}
    hopper = torch.cuda.get_device_capability(device) == (9, 0)
    fa3 = hopper and _importable("flash_attn_interface")
    if attention == "fa3" and not fa3:
        raise ValueError(
            "attention='fa3' requires a Hopper (sm_90) GPU and an importable flash_attn_interface"
        )
    dense = "fa3" if attention == "fa3" or (attention == "auto" and fa3) else "sdpa"
    kvcache = _select_kvcache_kernel()
    return {"prefill": dense, "decode": kvcache, "expert": kvcache}
