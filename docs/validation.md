<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoint validation

The results below record checks performed on 2026-10-07 and 2026-10-08. They
are historical measurements, not a fresh validation of every later code revision. The commands
use the supported PhysicalAI-AV input path with local or Hugging Face model
assets.

Those recorded runs precede the loader's ego-anchor correction. The current
loader uses the latest actual capture across all 22 images for its ego-history
window and coordinate frame, and reports that timestamp as `ego_t0_us`. The
requested `t0_us` still controls camera acquisition and sample identity. The
CPU regressions below cover the correction; the historical GPU measurements
remain associated with their recorded loader revisions.

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

The inputs are 33 packed samples from an internal evaluation set (not
distributed): 3 warm-up and 30 timed. Each has 22 images in two patch grids,
about 3,240 prompt tokens, and the 6.4-second ground-truth future, which is used
only for minADE. Each sample has a fixed seed for text sampling and diffusion
noise. Settings follow the CLI defaults: temperature 0.6, top-p 0.98, at most 256
generated tokens and ten Euler steps; greedy runs add `top_k=1`. Latency is the
synchronized model call (`sample_trajectories_from_data`) with inputs already on
the GPU, as p50 and p90 over the 30 timed samples. It excludes preprocessing, the
upload of the inputs and, for the ViT cache, `prepare()`. Calls that built graphs
are excluded.

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

The table lists the share of CoTs with identical tokens, and the mean XY distance
to the default trajectory for greedy one-CoT samples whose CoT is identical.
ΔminADE is the paired mean difference to the default path with a bootstrap 95%
interval. The default minADE is 2.792 m for greedy decoding with one CoT and
1.348 m with six CoTs. The rows are generated from the raw results.

| Option | Identical CoTs (greedy / sampled / six CoTs) | XY distance, same CoT | ΔminADE, greedy, one CoT | ΔminADE, six CoTs |
| --- | --- | --- | --- | --- |
| `--patch-embed linear` | 90.9% / 97.0% / 96.0% | 1.8 cm | +0.002 m [-0.006, +0.012] | -0.003 m [-0.012, +0.004] |
| ViT cache, eager | 100% / 100% / 100% | 0 (bitwise) | 0 | 0 |
| `--accel compile` | 93.9% / 97.0% / 94.9% | 2.4 cm | +0.010 m [-0.006, +0.034] | -0.000 m [-0.012, +0.010] |
| `--accel tuned` | 87.9% / 100% / 95.5% | 2.6 cm | +0.011 m [-0.006, +0.038] | -0.001 m [-0.012, +0.010] |
| `--accel tuned` with the ViT cache | 93.9% / 100% / 96.5% | 2.6 cm | +0.013 m [-0.003, +0.038] | -0.005 m [-0.026, +0.010] |
| All options | 90.9% / 100% / 96.0% | 2.3 cm | +0.009 m [-0.003, +0.026] | -0.008 m [-0.032, +0.007] |
| `--accel tuned`, FA3 | 87.9% / 100% / 94.9% | 3.3 cm | +0.009 m [-0.011, +0.043] | -0.005 m [-0.028, +0.010] |
| All options, FA3 | 90.9% / 100% / 96.0% | 2.3 cm | +0.005 m [-0.007, +0.020] | -0.002 m [-0.015, +0.008] |

Every interval includes zero; with 33 samples, the intervals span up to 3% of
minADE. One sample (ADE 7.4 m) dominates the positive one-CoT means. Its CoT is
unchanged, but its trajectory reacts to any numerical change: the GEMM patch
embedding alone moves it by 10 cm on average, and acceleration by 16–46 cm.
Without it, the one-CoT means lie between -0.005 and +0.003 m. No accelerated
output was bitwise identical to the default path, whereas the eager ViT cache
reproduced all 33 samples bit for bit under all three protocols.

With the compiled vision graphs, the ViT cache changes outputs relative to
`--accel tuned` alone: 93.9% of greedy CoTs, all sampled one-CoT CoTs and 96.0%
of six-CoT CoTs stay identical. Same-CoT trajectories move by 0.5 cm (0.8 cm with
six CoTs), and greedy minADE changes by +0.002 m [-0.002, +0.006].

These 33 samples are the only parity check of the SDPA configurations
(`--accel compile`, and `--accel tuned` without FlashAttention 3), which the
release environment uses. The larger evaluation of an earlier prototype
([below](#earlier-evidence)) covers `--accel tuned` with FlashAttention 3, with
no measurable change on 2,643 samples. In that evaluation, the `compile`
configuration (default-mode decode, SDPA vision and prefill) changed greedy
minADE by a small but significant +0.2%.

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
| Default, release environment | 952 / 1,153 | 1,099 / 1,283 |
| `--accel compile` | 141 / 163 (6.8x) | 506 / 532 (2.2x) |
| `--accel tuned` | 127 / 150 (7.5x) | 488 / 522 (2.3x) |
| `--accel tuned` with the ViT cache | 115 / 134 (8.3x) | 489 / 517 (2.2x) |
| All options | 114 / 133 (8.3x) | 492 / 521 (2.2x) |
| Default, FA3 environment | 987 / 1,190 | 1,139 / 1,314 |
| `--accel tuned`, FA3 | 122 / 142 (8.1x) | 493 / 525 (2.3x) |
| `--accel tuned`, FA3, with the ViT cache | 110 / 133 (8.9x) | 482 / 509 (2.4x) |
| All options, FA3 | 109 / 129 (9.0x) | 484 / 511 (2.4x) |

With six CoTs, vision and decoding are graphed. The prefill runs eagerly into
the static cache, and the expert graph replays only when all six CoTs have the
same length, which was true for 5 of the 33 inputs.

The default path is host-bound, and its p50 varied by up to 7% between runs
(945–1,008 ms). That masks small differences between sequential runs, so the eager
ViT cache was measured with alternating uncached and cached calls on the same
inputs. It saved 33 ms per call: the vision tower took 24 ms instead of 58 ms.
With the compiled graphs, the cache saved 13 ms per call (vision graphs: 7 ms
instead of 22 ms). `prepare()` itself is not in these numbers; it took about
3 ms per call in steady state on a loaded 16-core host. The GEMM patch embedding
made no measurable difference on the H100, where BF16 Conv3d is fast.

| Memory (GB), one CoT / six CoTs | Peak allocated | Peak reserved |
| --- | --- | --- |
| Default path | 5.64 / 8.61 | 5.92 / 13.36 |
| `--accel tuned` | 5.91 / 8.75 | 6.71 / 10.18 |
| `--accel tuned` with the ViT cache (256 frames) | 6.53 / 9.36 | 7.59 / 11.03 |

Reserved memory includes the graphs' private memory pools and the caching
allocator's free blocks. With acceleration, the static KV cache (0.41 GB per
CoT for a 3,584-token bucket) and the graphs stay allocated between calls.

From an empty compiler cache, building the eight graphs for one and six CoTs took
121 s with `--accel compile`, 170 s with `--accel tuned`, and 184 s with FA3;
the ViT cache adds two vision graphs (190 s in total). The first one-CoT call
alone took about 110 s with `--accel tuned`. With a populated cache, building six
of the graphs took 23 s.

### Reproducibility

The default path is deterministic: two runs on different nodes produced
bitwise-identical outputs for all 99 calls (33 samples, three protocols).
Compiled kernels are chosen when they are compiled, partly by benchmarking
candidates, and `--accel tuned` also autotunes the decode step. Repeated runs of
the same code with the same order of graph builds reproduced the accelerated
outputs bit for bit, including on another node. Building the graphs in a
different order can change that selection: in a test where only the point at
which the six-CoT expert graph was captured differed, a fresh compile changed 31
of 33 six-CoT trajectories in the last bits and 2 of 198 CoTs, and reusing the
other run's compiler cache reproduced its outputs bit for bit.

### Default path and CLI

On one raw internal sample, the CLI produced byte-identical
`output.safetensors`, `packed.safetensors` and `prompt.txt` with this change
and with `main`, for one and six CoTs on the H100 and for one CoT in FP32 on CPU.
The reports differ only in elapsed time. With the CLI's sampling settings,
`--accel tuned` generated the same CoT tokens as the default path, with every
trajectory point within 1.3 cm. Its `report.json` recorded five graph builds and
no fallbacks. With six CoTs, five of six CoTs matched, and the six-CoT expert
graph was captured during the warm-up run. In the release environment without
FlashAttention 2, `--accel tuned` stopped with an error before loading the
model, and `--accel tuned --accel-allow-partial` ran with only the vision tower
accelerated after a `RuntimeWarning` that named the eager parts.

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

The test suite passed 110 tests on CPU, skipping the four GPU tests and the two
private-bundle tests, and 114 tests on the H100 in both environments. New tests
cover the GEMM patch embedding against the Conv3d, kernel selection per GPU, the
static cache's `DynamicCache` behavior, the CPU attention reference, vision
grouping against the Transformers tower, accelerated sampling and anchor
completion against the eager model on a tiny Qwen3-VL, eager fallbacks, warm-up,
configuration checks, the order in which the options are applied, the ViT
cache's reuse, keys, fingerprint and eviction, the ViT cache on the accelerated
vision tower, and the new CLI options. On the GPU, the tests compare compiled
CUDA graphs with eager execution; check that replays after a build on another
input read every new input (bitwise equal to graphs built on that input); and
check that the graphed sampler draws the same random numbers as eager sampling
for one and three CoTs.

### Earlier evidence

The same techniques were first evaluated on this checkpoint in an earlier
prototype implementation, with Transformers 4.57.6 on H100, before this port. On
2,643 evaluation samples with six CoTs each, the tuned configuration
(FlashAttention 3 for vision and prefill, autotuned decode) changed minADE by
-0.0021 m [-0.0048, +0.0004] and 94% of CoTs were identical. A greedy one-CoT
run with all four graphs active changed it by -0.00003 m [-0.0021, +0.0020],
with 96.3% identical CoTs. Where a greedy CoT differed, the eager model's two best
logits at the first differing token were within one or two BF16 ulps. The
`compile` configuration (default-mode decode, SDPA vision and prefill) changed
six-CoT minADE by -0.0011 m [-0.0041, +0.0016] and greedy minADE by +0.0041 m
[+0.0003, +0.0093], or +0.2%. One CoT took 133.9 ms instead of 1,216 ms
(117.5 ms with the ViT cache). On DGX Spark (GB10, PyTorch 2.9.1, GEMM patch
embedding in both arms), it took 749 ms instead of 1,615 ms. These numbers come
from a different pipeline and Transformers version; they are context for the
measurements above, not a substitute for them.

## CPU tests

Install the development and dataset dependencies, then run the CPU suite:

```bash
uv sync --locked --extra dataset --dev
CUDA_VISIBLE_DEVICES="" uv run --locked pytest
```

Tests cover noncausal expert attention, shared prompt prefill, independent
reasoning candidates, cache reuse, anchor completion, left padding, HF save/load,
in-memory input validation, and CLI reports. Dataset tests cover per-camera
sampling, timestamp jitter and image association, the latest-capture ego anchor
across cameras, full rigid pose transforms at that anchor, shifted history bounds
including inclusive endpoints, anchor provenance in output records, missing
optional dependencies, and manifest navigation
selection and overrides. They use test fixtures and mocked dataset access; an
actual checkpoint and dataset execution is checked with the navigation command
above.
