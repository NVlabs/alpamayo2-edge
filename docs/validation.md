<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoint validation

## EA repository download and inference

On 2026-10-07, inference succeeded after downloading directly from
[nvidia/EA-Alpamayo2-Edge](https://huggingface.co/nvidia/EA-Alpamayo2-Edge/tree/b9d768920e925d8088afb2c628aa33acc3d36576),
revision `b9d768920e925d8088afb2c628aa33acc3d36576`. Both the HF Hub cache and Xet
cache started empty. The CLI received the repository ID and immutable revision;
its report resolves to the new EA snapshot. All 14 downloaded model assets,
including both complete weight shards, matched the uploaded SHA256 hashes.

The input was the [public PhysicalAI-AV navigation example](#public-physicalai-av-navigation-example):
clip `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4` at 4.0 seconds, with
`Turn right in 30m`, 22 camera frames, and 20 historical ego poses. On H100 80GB
with BF16 and SDPA, six reasoning continuations each produced one trajectory.
XYZ and rotations were finite, shaped `[1, 6, 1, 64, 3]` and
`[1, 6, 1, 64, 3, 3]`. Loading reported no missing, unexpected, or mismatched
keys and no errors. The runtime, dataset revision, and sampling settings match
the public-example table below. This verifies downloading, loading, and inference;
it does not establish benchmark quality or tuned performance.

For this same public clip, every packed input tensor, generated token ID, and
predicted XYZ/rotation tensor is exactly equal to the earlier run using the
original local checkpoint. This checks that the release config migration
preserves the observed inference result for this example.

The published configuration uses `Alpamayo2Edge`, `alpamayo2_edge`, and
`alpamayo2_edge_expert`, with standalone Hydra targets and stale backbone paths
removed. All 193 numerical/Boolean config values are unchanged. Weight shards,
tokenizer, processors, generation config, and chat template are byte-identical
to the original checkpoint. HF contains those model assets and a model card
template; examples and reference outputs were not uploaded.

Local evidence is in `.work/ea-download-validation/preflight.json`,
`download-audit.json`, and `output/report.json`. The upload and full download hash
audits are `.work/ea-release/upload-result.json` and
`.work/ea-release/fresh-download-sha256.json`; the before/after comparison is
`.work/ea-release/fresh-vs-original-public-nav.json`. Use the
[README example](../README.md#run-a-physicalai-av-navigation-example) to reproduce.

## Original checkpoint bundle validation

The original checkpoint bundle's complete model ran successfully on all ten
supplied examples with both SDPA and FlashAttention 2 on an **NVIDIA H100 80GB
HBM3**. Every run generated six
reasoning continuations, one expert trajectory per continuation, and 64 waypoints
per trajectory. All output positions and rotations were finite and had shapes
`[1, 6, 1, 64, 3]` and `[1, 6, 1, 64, 3, 3]` respectively.

The source for these historical comparisons is `nvidia/Internal-Alpamayo2-Edge`,
checkpoint step **6530**, bundle `2b/1002_model1_v3_plain_vqa1000_step6530`, HF revision
`7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4`. The runtime used PyTorch 2.8.0+cu128,
Transformers 5.17.0, and BF16. Settings followed the manifest: seed 43,
temperature 0.6, top-p 0.98, no top-k limit, maximum generation length 256, and
ten Euler diffusion steps.

### Measured results

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

## Reproduce the original bundle comparisons

These historical commands require separate access to the original private bundle
identified above, including its raw examples and reference outputs. Set
`EDGE_BUNDLE` to that local bundle directory, then run from the repository root.
The EA HF repository contains model assets only; for a directly downloadable
inference example, use the [public clip instructions](../README.md#run-a-physicalai-av-navigation-example).

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

## Public PhysicalAI-AV navigation example

On 2026-10-07, the CLI also ran end to end from the public PhysicalAI-AV SDK on
the [released Super navigation demo](https://github.com/NVlabs/alpamayo2/blob/59776920bbd981f2c989213bba5ee8594e71ed1d/examples/two_gpu_nav_cfg_demo.py).
The checked-in `examples/nav_samples.json` combines that demo's published clip
and time with its supplied navigation instruction:

| Field | Tested value |
| --- | --- |
| Dataset | `nvidia/PhysicalAI-Autonomous-Vehicles` |
| Dataset revision | `33f9bf447ed3bcb7d545ce13f4226f824214fafb` |
| Clip | `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4` |
| Clip-relative anchor | 4,000,000 microseconds |
| Navigation | `Turn right in 30m` |
| Input profile | 22 actual camera images, 20 historical ego poses |
| Runtime | H100 80GB, BF16, PyTorch 2.8.0+cu128, Transformers 5.17.0, SDPA |
| Dataset dependencies | physical-ai-av 0.2.0, av 19.0.1, pandas 3.0.6, pyarrow 25.0.1 |

The final run used the dataset dependencies resolved in `uv.lock`. All six
reasoning/trajectory sets completed with finite XYZ and rotations, shaped
`[1, 6, 1, 64, 3]` and `[1, 6, 1, 64, 3, 3]`. Checkpoint loading reported no
missing, unexpected, or mismatched parameters. The saved fused prompt contains
all 22 relative timestamp labels and the selected navigation instruction.
No future trajectory was loaded as an input. This is an execution check of one
published navigation example, not a navigation-quality metric or a comparison
with Super's classifier-free guidance outputs.

Follow the [public clip setup instructions](../README.md#run-a-physicalai-av-navigation-example),
then run the published EA checkpoint with the same sampling settings:

```bash
python -m alpamayo2_edge.inference \
    --model-id nvidia/EA-Alpamayo2-Edge \
    --revision b9d768920e925d8088afb2c628aa33acc3d36576 \
    --manifest examples/nav_samples.json \
    --num-traj-sets 6 --num-traj-samples 1 \
    --attn-implementation sdpa --output outputs/physicalai-nav
```

Local evidence is saved under `.work/results-physicalai-nav-sdpa/`, including
`report.json`, the actual packed model inputs, fused prompt, and outputs. The
report records the dataset revision and decoder dependency versions. Dataset
revisions can be selected through a supplied `PhysicalAIAVDatasetInterface` in
the Python API; the CLI resolves the SDK's default dataset revision at run time.

## Inference acceleration

On 2026-10-08, the opt-in acceleration described in
[acceleration.md](acceleration.md) was compared with the default path on the
same inputs and seeds.

### Inputs and settings

The inputs are 33 packed samples of the 1002 checkpoint's internal evaluation
suite `cot_public_golden`: 3 warm-up and 30 timed. Each has 22 images in two
patch grids, about 3,240 prompt tokens, and the 6.4-second ground-truth future,
which is used only for minADE. These samples are not distributed. Each sample
has a fixed seed for text sampling and diffusion noise. Settings follow the CLI
defaults: temperature 0.6, top-p 0.98, at most 256 generated tokens and ten Euler
steps; greedy runs add `top_k=1`. Latency is the synchronized model call
(`sample_trajectories_from_data`) with inputs already on the GPU, as p50 and p90
over the 30 timed samples. Calls that built graphs are excluded.

### Environments

Both environments ran on one node of NVIDIA H100 80GB HBM3 GPUs:

| | Release environment | FlashAttention 3 environment |
| --- | --- | --- |
| Packages | `uv.lock` (PyTorch 2.8.0+cu128, Transformers 5.17.0, Triton 3.4.0) plus the prebuilt FlashAttention 2.8.3 wheel | PyTorch 2.8.0+cu128, Transformers 5.17.0, Triton 3.4.0, FlashAttention 2.8.3 and 3.0.0 |
| Attention in the graphs | SDPA for vision and prefill; FA2 for decode and the expert | FA3 for vision and prefill; FA2 for decode and the expert |

The FlashAttention 2 wheel needs glibc 2.32 or newer, so the release environment
ran in an Ubuntu 22.04 userspace. Each option was compared with the default path
of its own environment; the two default paths produced bitwise-identical outputs
for all samples.

### Parity on H100

The table lists the share of CoTs with identical tokens and the mean XY distance
to the default trajectory where the CoT is identical. ΔminADE is the paired mean
difference to the default path with a bootstrap 95% interval. The default
minADE is 2.792 m for greedy decoding with one CoT and 1.348 m with six CoTs.

| Option | Identical CoTs (greedy / sampled / six CoTs) | XY distance, same CoT | ΔminADE, greedy, one CoT | ΔminADE, six CoTs |
| --- | --- | --- | --- | --- |
| `--patch-embed linear` | 90.9% / 97.0% / 96.0% | 1.8 cm | +0.002 m [-0.007, +0.012] | -0.003 m [-0.012, +0.004] |
| ViT cache, eager | 100% / 100% / 100% | 0 (bitwise) | 0 | 0 |
| `--accel compile` | 93.9% / 97.0% / 94.9% | 2.5 cm | +0.010 m [-0.006, +0.034] | -0.000 m [-0.012, +0.011] |
| `--accel tuned` | 87.9% / 97.0% / 97.0% | 2.7 cm | +0.011 m [-0.006, +0.036] | -0.000 m [-0.013, +0.011] |
| `--accel tuned` with the ViT cache | 93.9% / 97.0% / 97.0% | 2.6 cm | +0.011 m [-0.006, +0.036] | -0.005 m [-0.027, +0.010] |
| All options | 87.9% / 100% / 96.5% | 2.5 cm | +0.008 m [-0.005, +0.026] | -0.008 m [-0.032, +0.007] |
| `--accel tuned`, FA3 | 87.9% / 100% / 95.5% | 3.3 cm | +0.012 m [-0.009, +0.046] | -0.006 m [-0.029, +0.010] |
| All options, FA3 | 87.9% / 97.0% / 96.0% | 2.2 cm | +0.004 m [-0.009, +0.019] | -0.002 m [-0.015, +0.008] |

Every interval includes zero; with 33 samples, the intervals span 1–3% of
minADE. One sample (ADE 7.4 m) dominates the positive one-CoT means. Its CoT is
unchanged, but its trajectory reacts to any numerical change: the GEMM patch
embedding alone moves it by 10 cm on average, and acceleration by 35 cm. The
other samples' differences are centered near zero. No accelerated output was
bitwise identical to the default path, whereas the eager ViT cache reproduced
all 33 samples bit for bit under all three protocols.

With the compiled vision graphs, the ViT cache changes outputs relative to
`--accel tuned` alone: 93.9% of greedy CoTs, all sampled one-CoT CoTs and 98% of
six-CoT CoTs stay identical. Same-CoT trajectories move by 0.6 cm, and greedy
minADE changes by -0.0001 m [-0.006, +0.005].

### Latency on H100

Each environment ran its options back to back in one process on an otherwise
idle node, on one model instance. Accelerated options first ran one untimed
pass over all inputs, after their graphs were built. The table lists p50 and
p90 over the 30 timed samples in milliseconds, with the speedup of the p50 over
the same environment's default path. The ViT cache rows measure streaming
steady state: past frames come from an earlier encoding, and the six newest
frames are encoded in the timed call.

| Option | One CoT, p50 / p90 | Six CoTs, p50 / p90 |
| --- | --- | --- |
| Default, release environment | 945 / 1,138 | 1,080 / 1,246 |
| `--accel compile` | 140 / 163 (6.7x) | 482 / 509 (2.2x) |
| `--accel tuned` | 128 / 148 (7.4x) | 484 / 510 (2.2x) |
| `--accel tuned` with the ViT cache | 115 / 135 (8.2x) | 476 / 510 (2.3x) |
| All options | 115 / 134 (8.2x) | 475 / 507 (2.3x) |
| Default, FA3 environment | 991 / 1,194 | 1,134 / 1,304 |
| `--accel tuned`, FA3 | 123 / 142 (8.1x) | 494 / 520 (2.3x) |
| `--accel tuned`, FA3, with the ViT cache | 111 / 130 (8.9x) | 488 / 515 (2.3x) |
| All options, FA3 | 110 / 129 (9.0x) | 487 / 516 (2.3x) |

With six CoTs, only vision and decoding are graphed. The prefill runs eagerly
into the static cache, and the expert runs eagerly unless all six CoTs have the
same length, which was true for 5 of the 33 inputs.

The default path is host-bound, and its p50 varied by up to 7% between runs
(945–1,008 ms). That masks small differences between sequential runs, so the eager
ViT cache was measured with alternating uncached and cached calls on the same
inputs. It saved 33 ms per call: the vision tower took 24 ms instead of 58 ms.
With the compiled graphs, the cache saved 13 ms per call (vision graphs: 7 ms
instead of 22 ms). The GEMM patch embedding made no measurable difference on the
H100, where BF16 Conv3d is fast.

Peak allocated memory was 5.64 GB for one CoT and 8.61 GB for six CoTs on the
default path, 5.91 and 8.75 GB with `--accel tuned`, and 6.53 and 9.36 GB with the
ViT cache added. The cache held 256 frames. From an empty compiler cache, building
the eight graphs for one and six CoTs took 123 s with `--accel compile`, 170 s with
`--accel tuned`, and 191 s with FA3; the ViT cache adds two vision graphs (193 s
in total). With a populated cache, building six of the graphs took 23 s.

### Default path and CLI

On one raw internal sample, the CLI produced byte-identical
`output.safetensors`, `packed.safetensors` and `prompt.txt` with this change
and with `main`, for one and six CoTs on the H100 and for one CoT in FP32 on CPU.
The reports differ only in elapsed time. With the CLI's sampling settings,
`--accel tuned` generated the same CoT tokens as the default path, with
trajectories within 1.6 cm (6.1 cm with FA3). Its `report.json` recorded five
graph builds and no fallbacks. With six CoTs, five of six CoTs matched.

### CPU checks on the real checkpoint

`AccelConfig(compile=False, cuda_graphs=False)` runs the accelerated code without
compilation or graphs, with PyTorch reference attention. On packed samples from
the inputs above, in FP32 on CPU, compared with the default path:

| Check | Result |
| --- | --- |
| Greedy, one CoT | Tokens identical (24 CoT tokens and the anchor); max XYZ difference 9.5e-7 m |
| Greedy, two CoTs | Tokens identical; max XYZ difference 3.8e-6 m |
| CLI sampling settings, three CoTs, two samples | All tokens identical; trajectories bitwise equal |
| ViT cache: 16 frames reused, 6 encoded | Tokens identical; trajectories bitwise equal |

These checks exercise the port to Transformers 5.17 with the real shapes: two
patch grids, three DeepStack levels, MRoPE positions, the shared prefill, the
sampler's random draws, and the expert reading the static cache, both graphed
and through the eager fallback for CoTs of different lengths. They do not
cover compiled kernels or CUDA graphs.

The test suite passed 104 tests on CPU, skipping the GPU test and the two
private-bundle tests, and 105 tests on the H100. New tests cover the GEMM patch
embedding against the Conv3d, kernel selection per GPU, the static cache's
`DynamicCache` behavior, the CPU attention reference, vision grouping against
the Transformers tower, accelerated sampling and anchor completion against the
eager model on a tiny Qwen3-VL, eager fallbacks, warm-up, the ViT cache's reuse,
keys and eviction, and the new CLI options. The GPU test compares compiled CUDA
graphs with eager execution on the tiny model.

### Earlier evidence

The same techniques were first evaluated on this checkpoint in NVIDIA's internal
Alpamayo training code, with Transformers 4.57.6 on H100, before this port. On
2,643 evaluation samples with six CoTs each, the tuned configuration changed
minADE by -0.0021 m [-0.0048, +0.0004] and 94% of CoTs were identical. A greedy
one-CoT run with all four graphs active changed it by -0.00003 m
[-0.0021, +0.0020], with 96.3% identical CoTs. Where a greedy CoT differed, the
eager model's two best logits at the first differing token were within one or two
BF16 ulps. Without FlashAttention 3, the greedy change was +0.0041 m
[+0.0003, +0.0093], or +0.2%. One CoT took 133.9 ms instead of 1,216 ms (117.5 ms
with the ViT cache). On DGX Spark (GB10, PyTorch 2.9.1, GEMM patch embedding in
both arms), it took 749 ms instead of 1,615 ms. These numbers come from a
different pipeline and Transformers version; they are context for the
measurements above, not a substitute for them.

## CPU tests

The full suite passed **78 tests**, including the optional real-bundle input
comparison and timestamp perturbation test. Tests cover noncausal expert
attention, cache reuse, CoT/sample counts, anchor completion, left padding,
HF save/load, input validation, and CLI reports. Public dataset tests additionally
cover per-camera sampling, timestamp jitter and image association, full rigid
pose transforms, inclusive time bounds, missing optional dependencies, and
manifest navigation selection/overrides.
Ruff checks, formatting, lockfile validation, and source/wheel builds also passed.

Run `CUDA_VISIBLE_DEVICES="" uv run --locked pytest` for the CPU suite. The saved
fixture comparison and processor timestamp test require the original private
bundle (these fixtures are not in the EA HF repository):

```bash
CUDA_VISIBLE_DEVICES="" ALPAMAYO2_EDGE_TEST_BUNDLE="$EDGE_BUNDLE" \
    uv run --locked pytest
```

The optional tests compare every packed field for `sample_000` against its saved
reference and verify that timestamp changes affect the actual processor output.
The all-ten-example input comparisons above come from the GPU replay artifacts
and the CPU generation-boundary audit, without synthetic image fixtures.
