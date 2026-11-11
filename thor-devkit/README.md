<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Alpamayo 2 Edge on the Jetson AGX Thor Developer Kit

This directory is a separate uv project that locks a Thor (aarch64, sm_110,
CUDA 13.4) environment for the same `alpamayo2_edge` source in the parent
directory. The repository root keeps its x86_64 lock (PyTorch 2.8.0, CUDA 12,
FlashAttention 2); this project overrides those pins:

| Component | Root (`uv.lock`) | Thor (`thor-devkit/uv.lock`) |
| --- | --- | --- |
| PyTorch / torchvision | 2.8.0 / 0.23.0 | 2.15.0+cu134 / 0.29.2+cu134 ([cu134 index](https://download.pytorch.org/whl/cu134)) |
| KV-cache attention for `--accel` | FlashAttention 2 (`fa2`) | FlashAttention 4 (`flash-attn-4`, `fa4`) |
| Extras | `dataset`, `notebook` optional | `dataset`, `notebook`, FlashInfer 0.7.3 (sm110a) |

The lock resolves only for `linux`/`aarch64`, and `alpamayo2-edge` is installed in
editable mode from `..`, so source changes in the repository apply directly.
`requirements-frozen.txt` is the `uv pip freeze` of the validated environment; the
lock constrains every package to those versions.

## Setup

From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --locked --project thor-devkit
source .venv/bin/activate
```

Use an absolute `UV_PROJECT_ENVIRONMENT`; a relative path resolves against
`thor-devkit/`. Do not add `thor-devkit` to a `[tool.uv.workspace]` in the root
`pyproject.toml`: a workspace is solved as one, and the root's `torch==2.8.0` pin
then conflicts with this project's `torch==2.15.0+cu134`.

After changing `pyproject.toml` here, relock with `uv lock --project thor-devkit`.

## Run

The commands in the [main README](../README.md) apply unchanged, for example:

```bash
export EDGE_MODEL="nvidia/Alpamayo2-Edge"
TRITON_DEFAULT_FP_FUSION=1 python -m alpamayo2_edge.inference \
    --model-id "$EDGE_MODEL" \
    --manifest examples/nav_samples.json \
    --accel tuned \
    --output outputs/accelerated
```

`report.json` lists `"decode": "fa4"` and `"expert": "fa4"` under
`acceleration.kernels`. The first `--accel tuned` run spends about a minute
compiling and capturing graphs. FlashAttention 4 prints `aux_data ... cannot be
converted to a JitArgument` warnings from the CuTe DSL; they are harmless.
