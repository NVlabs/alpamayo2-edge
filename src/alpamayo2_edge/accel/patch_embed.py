# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vision patch embedding as one GEMM instead of a stride == kernel Conv3d."""

import types

import torch
import torch.nn.functional as F


def _linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``Qwen3VLVisionPatchEmbed.forward`` as one GEMM.

    The Conv3d kernel equals its stride, so it produces one output per patch: the
    dot product of each flattened ``[C*T*P*P]`` patch row with each flattened
    filter, in the same element order as the original row-major views.
    """
    weight = self.proj.weight
    patches = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    return F.linear(patches, weight.reshape(weight.shape[0], -1), self.proj.bias)


def use_linear_patch_embedding(model: torch.nn.Module) -> None:
    """Replace the vision patch embedding's Conv3d by the equivalent GEMM (in place).

    Results agree with the convolution to rounding (about one BF16 ulp); the GEMM
    avoids slow BF16 Conv3d fallbacks, such as PyTorch 2.9 on GB10 (sm_121).
    """
    embed = model.vlm.model.visual.patch_embed
    conv = embed.proj
    if not (
        isinstance(conv, torch.nn.Conv3d)
        and tuple(conv.kernel_size) == tuple(conv.stride)
        and tuple(conv.padding) == (0, 0, 0)
        and tuple(conv.dilation) == (1, 1, 1)
        and conv.groups == 1
    ):
        raise ValueError(f"Patch embedding is not a kernel == stride Conv3d: {conv}")
    embed.forward = types.MethodType(_linear_patch_embed_forward, embed)
