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

## Chat template and timestamp audit

The follow-up audit on 2026-10-06 cross-referenced the internal Alpamayo / Alpax
sources and the actual 1002 training run, rather than assuming the generic Alpax
export defaults were the checkpoint's recipe. The run's resolved
`evaluations.cot_public_golden` config selects the Edge conversation factory with
`components_order=[image, traj_history, nav_instruction, prompt]`,
`components_prompt=[cot, traj_future]`, and `generation_mode=true`.

The Edge conversation source in the training `code.tar.gz`, checkpoint-pinned
[runtime dcce725](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/dcce725763730554aaad1337aed88db5132fcadc/src/alpamayo/data/chat_template/conversation_alpamayo_2_edge.py),
and audited [main 1fb0b3f](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/1fb0b3f2416effe64bad6b6b032cab0352d5873c/src/alpamayo/data/chat_template/conversation_alpamayo_2_edge.py)
is byte-identical. Its SHA-256 is
`b85bb257f84746edf6006b0bd6ee56c819f18207937e1da93bd3b205ef59dcaa`, matching
the bundle's loaded-source hash. These links require internal GitLab access.

Executing both fetched internal versions with the resolved recipe produced the
same rendered prompts as this package for all ten samples, plus matching results
for five system-prompt precedence cases. This comparison used the samples'
actual timestamps/navigation and indexed tiny images to check image ordering;
it did not read the saved reference prompts. Internal camera-specific image
bounds were resolved before comparing message structures. Actual full-sized
image preprocessing is covered by the separate generation-boundary audit below.

The standalone recipe preserves camera labels, timestamp/image pairing, 57
history placeholders, component newlines, `Navigation input is: `, and
`Your task: `. It emits system and user messages and lets the saved Qwen3 Jinja
template append the assistant generation prefix. It does not prefill a CoT
wrapper. The pinned internal datapacker passes `enable_thinking=false`; the
saved Jinja contains no branch using that option, so omitting it does not change
the rendered prompt. No change to the saved Jinja or conversation recipe is
required for this checkpoint.

The timestamp path was exercised on CPU through raw sample loading, joint
camera/time sorting, the actual checkpoint processor, history-token fusion, and
the production `sample_trajectories_from_data` method. The audit intercepted its
`vlm.generate` call before a forward pass. On **10/10 examples**, the intercepted
token IDs, attention masks, image grids, and pixel values exactly match the saved
references; all decoded timestamp labels and image-token type spans also match.
This verifies the VLM's input boundary without claiming an additional GPU run.

The zero point is the final camera-major timestamp, which is the latest camera-6
capture in this profile. Across the ten examples it precedes the global latest
capture by 0–33,380 microseconds and `event_t0_us` by 1–33,380 microseconds. This
matches the internal implementation. Subtraction happens in `int64` before
conversion to seconds, preserving precision for epoch timestamps. Labels round
to one decimal place and normalize negative zero; the model receives these
labels as text tokens immediately before each image.

Controlled changes to `sample_000` confirmed the connection to the model input:

- Moving raw frame 5's capture time 160 ms earlier changes `<t=-1.0s>` to
  `<t=-1.2s>` and exactly one token ID at the generation boundary. Pixels, grids,
  and multimodal token types stay equal.
- Moving that capture after the other camera-0 frames reorders the processor's
  image patch chunks from `[0, 1, 2]` to `[1, 2, 0]`, with the corresponding labels.
- Adding 50 seconds to all captures leaves the generation input IDs unchanged.

The detailed local evidence is `.work/chat-template-audit-v2.json` and
`.work/timestamp-audit-v2.json`.
`tests/test_timestamps.py` retains regressions for epoch precision, origin
invariance, vector/column input shapes, negative zero, the last-camera zero
point, actual processor token changes, and image patch reordering. The checks
establish correct delivery of timing information; they do not measure how
strongly the learned model uses it. Changes within the same 0.1-second rounding
interval can intentionally produce identical token IDs.

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

The full suite passed **52 tests**, including the optional real-bundle input
comparison and timestamp perturbation test. Tests cover noncausal expert
attention, cache reuse, CoT/sample counts, anchor completion, left padding,
HF save/load, input validation, and CLI reports.
Ruff checks, formatting, lockfile validation, and source/wheel builds also passed.

Run `CUDA_VISIBLE_DEVICES="" uv run --locked pytest` for the CPU suite. The saved
fixture comparison and processor timestamp test require the downloaded private bundle:

```bash
CUDA_VISIBLE_DEVICES="" ALPAMAYO2_EDGE_TEST_BUNDLE="$EDGE_BUNDLE" \
    uv run --locked pytest
```

The optional tests compare every packed field for `sample_000` against its saved
reference and verify that timestamp changes affect the actual processor output.
The all-ten-example input comparisons above come from the GPU replay artifacts
and the CPU generation-boundary audit, without synthetic image fixtures.
