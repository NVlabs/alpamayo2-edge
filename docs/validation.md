<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoint validation

The complete model ran successfully on all ten supplied examples with both SDPA
and FlashAttention 2 on an **NVIDIA H100 80GB HBM3**. Every run generated six
reasoning continuations, one expert trajectory per continuation, and 64 waypoints
per trajectory. All output positions and rotations were finite and had shapes
`[1, 6, 1, 64, 3]` and `[1, 6, 1, 64, 3, 3]` respectively.

The tested source is `nvidia/Internal-Alpamayo2-Edge`, checkpoint step **6530**,
bundle `2b/1002_model1_v3_plain_vqa1000_step6530`, HF revision
`7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4`. The runtime used PyTorch 2.8.0+cu128,
Transformers 5.17.0, and BF16. Settings followed the manifest: seed 43,
temperature 0.6, top-p 0.98, no top-k limit, maximum generation length 256, and
ten Euler diffusion steps.

## Measured results

| Check | SDPA | FlashAttention 2 |
| --- | --- | --- |
| All six trajectories produced with valid shapes and finite values | 10/10 examples | 10/10 examples |
| Packed inputs exactly equal to saved references | 10/10 examples | 10/10 examples |
| Generated token IDs exactly equal to saved references | 8/10 examples | 9/10 examples |
| Examples with different generated token IDs | `sample_003`, `sample_009` | `sample_009` |
| Largest absolute XYZ difference from reference across all examples | 0.890703 | 0.837526 |

Packed-input equality covers all six saved fields: token IDs, attention mask,
image grids, normalized image patches, history positions, and history rotations.
**Full output parity with the supplied references is not established.** Generated
tokens differ on the examples listed above, and trajectory values differ even on
examples whose generated tokens match. These are successful inference and exact
preprocessing checks, not a claim that all reference tolerances pass. The XYZ
differences include cases conditioned on different generated text.

A separate full-state audit found **947/947 trained parameter tensors exactly
equal** to their checkpoint tensors, with no dtype conversions, missing keys,
unexpected keys, mismatched keys, or nonfinite parameters. All **13 nonpersistent
buffers** matched their deterministic construction: six RoPE frequency buffers,
four action normalization scalars, and three Fourier frequency buffers. Input
embeddings and the language-model head remain distinct, as saved in the checkpoint.

Evidence was collected in `results-sdpa/report.json`, `results-fa2/report.json`,
`comparison.json`, `weight_parity_report.json`, and
`deterministic_buffer_report.json` under the local `.work/` directory. The runs
were correctness checks; no tuned performance benchmark was performed.

## Reproduce inference

Follow the [README download instructions](../README.md#download-the-checkpoint-and-examples)
to populate `EDGE_BUNDLE`, then run from the repository root:

```bash
uv sync --locked --dev
source .venv/bin/activate

python -m alpamayo2_edge.inference \
    --bundle "$EDGE_BUNDLE" --all-samples \
    --attn-implementation sdpa --output outputs/validation-sdpa

# Requires FlashAttention 2.8.3 installed in this environment.
python -m alpamayo2_edge.inference \
    --bundle "$EDGE_BUNDLE" --all-samples \
    --attn-implementation flash_attention_2 --output outputs/validation-fa2
```

The CLI records loading diagnostics, versions, settings, and per-example results
in each output root's `report.json`; each example also saves packed inputs and
output tensors. Reproducing these commands verifies execution and shape/finite
checks. Reference comparison is a separate operation on those saved artifacts.

## CPU tests

The full suite passed **48 tests**, including the optional real-bundle input
comparison. Tests cover noncausal expert attention, cache reuse, CoT/sample counts,
anchor completion, left padding, HF save/load, input validation, and CLI reports.
Ruff checks, formatting, lockfile validation, and source/wheel builds also passed.

Run `CUDA_VISIBLE_DEVICES="" uv run --locked pytest` for the CPU suite. The saved
fixture comparison is optional and requires the downloaded private bundle:

```bash
CUDA_VISIBLE_DEVICES="" ALPAMAYO2_EDGE_TEST_BUNDLE="$EDGE_BUNDLE" \
    uv run --locked pytest
```

That optional test compares every packed field for `sample_000` against its saved
reference. The all-ten-example input comparison above comes from the GPU replay
artifacts, and does not depend on a synthetic image fixture.
