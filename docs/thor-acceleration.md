<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference acceleration on the Jetson AGX Thor Developer Kit

This page covers what is specific to the NVIDIA Jetson AGX Thor Developer Kit
(aarch64, sm_110, CUDA 13.4). The accelerated parts, graph shapes, fallbacks and
output semantics are the same as on other GPUs; see
[acceleration.md](acceleration.md) for those.

| | x86_64 (root `uv.lock`) | Jetson AGX Thor (`thor-devkit/uv.lock`) |
| --- | --- | --- |
| PyTorch / CUDA | 2.8.0 / CUDA 12 | 2.15.0+cu134 / CUDA 13.4 |
| Prefill attention (`--accel-attention auto`) | SDPA, or FA3 on Hopper | SDPA |
| Decode and expert attention | FlashAttention 2 (`fa2`) | FlashAttention 4 (`fa4`) |
| `--attn-implementation flash_attention_2` | With the `flash` extra | Not available |

## Requirements

Install the locked Thor environment from the repository root, as described in
[thor-devkit/README.md](../thor-devkit/README.md):

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --locked --project thor-devkit --all-extras
source .venv/bin/activate
```

Its `flash` extra provides `flash-attn-4[cu13]==4.0.0b38`, which installs only
`flash_attn.cute`. FlashAttention 2 has no sm_110 build. Do not install it next to
FlashAttention 4: when both are importable, FlashAttention 2 is selected.

Check that the decode and expert parts resolve to FlashAttention 4:

```bash
python -c "from alpamayo2_edge.accel import resolve_kernels; print(resolve_kernels('auto', 'cuda'))"
# {'prefill': 'sdpa', 'decode': 'fa4', 'expert': 'fa4'}
```

If `decode` is `None`, `flash_attn.cute` failed to import (for example, a missing
`nvidia-cutlass-dsl`), and `--accel` stops with an error before loading the model.

## Usage

```bash
export EDGE_MODEL="nvidia/Alpamayo2-Edge"
TRITON_DEFAULT_FP_FUSION=1 python -m alpamayo2_edge.inference \
    --model-id "$EDGE_MODEL" \
    --manifest examples/nav_samples.json \
    --accel tuned \
    --output outputs/accelerated
```

`TRITON_DEFAULT_FP_FUSION=1` keeps Triton's current FP-fusion default and
silences its deprecation warning. It is optional.

Keep `--accel-attention auto` or `sdpa`. FlashAttention 3 is Hopper-only, so
`--accel-attention fa3` fails on Thor.

In `report.json`, the top-level `attention_backend` is `--attn-implementation`,
which is the attention used by the eager model (`sdpa`). The kernels used by the
graphs are under `acceleration`:

```json
"kernels": {"prefill": "sdpa", "decode": "fa4", "expert": "fa4"},
"fallbacks": {},
"flash_attn": "4.0.0b38"
```

## How FlashAttention 4 is used

FlashAttention 2's `flash_attn_with_kvcache` appends the new keys and values to
the static KV cache and attends the valid prefix in one call. FlashAttention 4 has
no fused cache append, so the `fa4` kernel takes two steps inside the same custom
op:

1. It writes the new keys and values into the cache at `cache_seqlens` using
   tensor indexing, with no host synchronization, so the write is captured in the
   CUDA graph.
2. It calls `flash_attn.cute.flash_attn_varlen_func` on the batched cache with
   `seqused_k = cache_seqlens + new_tokens`, which uses the same bottom-right
   causal alignment as FlashAttention 2.

Results match the PyTorch reference kernel within BF16 rounding, including after
a CUDA-graph replay with new sequence lengths.

## Compilation and warm-up

The first call with each shape compiles and captures its graphs. On the Thor
Developer Kit with `--accel tuned`, the example manifest built three graphs in
about 64 s: about 5 s for the prefill, 39 s for decoding and 20 s for the expert.
After that, the sample took about 2.7 s. FlashAttention 4 also JIT-compiles its
CuTe DSL kernels on first use, during the warm-up.

Inductor and Triton cache compiled kernels under `/tmp` by default. Set
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a persistent directory to
reuse them across reboots:

```bash
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/alpamayo2-edge/inductor"
export TRITON_CACHE_DIR="$HOME/.cache/alpamayo2-edge/triton"
```

## Known warnings

These do not affect results:

- `UserWarning: Argument aux_data (position 18) cannot be converted to a
  JitArgument ...` comes from the CuTe DSL while FlashAttention 4 compiles.
- `DeprecationWarning` about `TensorMapManager` or `TmemAllocator` comes from
  FlashAttention 4's use of `nvidia-cutlass-dsl`.
- `TRITON_DEFAULT_FP_FUSION is unset` comes from Triton 3.9; see
  [Usage](#usage).

## Jetson AGX Orin

The `thor-devkit` environment also runs on the Jetson AGX Orin 32GB Developer Kit
(sm_87) with JetPack 7.2.1 and its CUDA 13.2 driver, using the same commands.
`--accel tuned` graphs the prefill, decode and expert with `fa4` for decode and
expert attention and no fallbacks; FlashAttention 4 selects its Ampere (sm_8x)
kernels. As on Thor, the cu134 PyTorch wheels run on the 13.2 driver through CUDA
minor-version compatibility, without `cuda-compat`. The GPU tests in
`tests/test_accel.py` pass there as well.

## Tests

```bash
python -m pytest tests/test_accel.py -q
```

On Thor, the GPU tests (`*_on_gpu`) run the compiled CUDA graphs with the `fa4`
kernel. They are skipped when neither FlashAttention 2 nor 4 is importable.
