<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoint validation

The results below record checks of the released checkpoint performed on 2026-11-03. They
are measurements of that date, not a fresh validation of every later code revision. The
commands use the supported PhysicalAI-AV input path with local or Hugging Face model assets.

## Release assets and loading

The released configuration uses the standalone `alpamayo2_edge` model identifiers and
Hydra targets, with the training-time backbone path removed; every numerical and Boolean
configuration value equals the training checkpoint's. Weight shards and their index,
tokenizer, processor configuration, generation config, and chat template are
byte-identical to the training checkpoint.

On CPU, the release files passed `validate_checkpoint` and a strict FP32 `load_model` with
no missing, unexpected, or mismatched keys (2,651,163,378 parameters). The tokenizer has
135,102 tokens with `<unk>` (ID 0) as padding. With `enable_thinking=False`, the
checkpoint chat template ends the driving prompt with `<think></think>`, as in training.

## Public PhysicalAI-AV navigation example

The CLI ran end to end from the public PhysicalAI-AV SDK on the
[released Super navigation demo](https://github.com/NVlabs/alpamayo2/blob/59776920bbd981f2c989213bba5ee8594e71ed1d/examples/two_gpu_nav_cfg_demo.py).
The checked-in `examples/nav_samples.json` combines that demo's published clip and time
with its supplied navigation instruction:

| Field | Tested value |
| --- | --- |
| Dataset | `nvidia/PhysicalAI-Autonomous-Vehicles` |
| Dataset revision | `39414de6f0ab867238ad6a0299fcd18b26081b42` |
| Clip | `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4` |
| Clip-relative anchor | 4,000,000 microseconds |
| Navigation | `Turn right in 30m` |
| Input profile | 22 actual camera images, 20 historical ego poses |
| Runtime | H100 80GB, BF16, PyTorch 2.8.0+cu128, Transformers 5.17.0 |
| Dataset dependencies | physical-ai-av 0.2.0, av 19.0.1, pandas 3.0.6, pyarrow 25.0.1 |
| Sampling | seed 43, six reasoning continuations, one trajectory each, 10 diffusion steps |

The run was repeated with SDPA and with FlashAttention 2. Both reported no missing,
unexpected, or mismatched parameters and produced finite trajectories shaped
`[1, 6, 1, 64, 3]` and `[1, 6, 1, 64, 3, 3]`. The packed inputs of the two runs were
bitwise equal, all six reasoning texts were identical, and the trajectory endpoints
differed by at most 1.9 cm. Every continuation decided to turn right at the
intersection, for example "Decelerate and turn right at the intersection to follow our
route." The saved fused prompt contains all 22 relative timestamp labels and the
navigation instruction, and ends with `<think></think>`; no future trajectory was loaded
as an input. The model call took 4.0 seconds for six continuations with SDPA.

The same SDPA run, with the model downloaded from Hugging Face into empty Hub and Xet
caches, reproduced the packed inputs, generated token IDs, and trajectories bitwise.
Every downloaded file matched the release files' SHA256.

This is an execution check of one published navigation example, not a navigation-quality
metric. Follow the [public clip setup instructions](../README.md#run-a-physicalai-av-navigation-example)
to reproduce it:

```bash
python -m alpamayo2_edge.inference \
    --model-id nvidia/Alpamayo2-Edge \
    --revision <revision from the README> \
    --manifest examples/nav_samples.json \
    --num-traj-sets 6 --num-traj-samples 1 --seed 43 \
    --attn-implementation sdpa --output outputs/physicalai-nav
```

Each run's `report.json` records the dataset revision and decoder dependency versions.
Dataset revisions can be selected through a supplied `PhysicalAIAVDatasetInterface` in the
Python API; the CLI resolves the SDK's default dataset revision at run time.

## Inference acceleration

On the same H100 and example, `--accel tuned` was compared with the default path using
FlashAttention 2, after two untimed warm-up calls; the accelerated prompt prefill used
FlashAttention 3, and decode and expert attention FlashAttention 2. Packed inputs were
bitwise equal in both comparisons.

| Continuations | Generated tokens equal to the default path | Max endpoint difference | Model call, default → accelerated |
| --- | --- | --- | --- |
| 1 | 1 of 1 | 1.4 cm | 3.79 s → 1.48 s |
| 6 | 6 of 6 | 2.2 cm | 3.95 s → 1.94 s |

With six continuations the prefill ran eagerly, and so did the expert, because the
continuations differed in length. This covers one example and seed; it is not a latency
benchmark or a quality measurement.

## Course-heading action space

`UnicycleAccelCurvatureCourseActionSpace` was compared with the training implementation
on CPU in FP32: encoding, `t0` states and decoding were bitwise equal for 12 test
scenarios and for 1,000 random trajectories, including histories stored in BF16. Decoding
covered encoded futures and arbitrary actions decoded from the full history, as inference
does. The checkpoint's configuration instantiates both action spaces (the future
trajectory tokenizer's and the expert's) as the course-heading class with every saved
parameter, including `slip_gradient` 2.5e-3 and `theta_lambda` 3e-6.

## CPU tests

Install the development and dataset dependencies, then run the CPU suite:

```bash
uv sync --locked --extra dataset --dev
CUDA_VISIBLE_DEVICES="" uv run --locked pytest
```

Tests cover noncausal expert attention, shared prompt prefill, independent
reasoning candidates, cache reuse, anchor completion, left padding, HF save/load,
in-memory input validation, the driving prompt's closed think block, patch normalization
against the checkpoint processor, and CLI reports. Dataset tests cover per-camera
sampling, timestamp jitter and image association, ego-frame pose transforms,
inclusive time bounds, missing optional dependencies, and manifest navigation
selection and overrides. They use small randomly initialized models, test fixtures and
mocked dataset access; an actual checkpoint and dataset execution is checked with the
navigation command above.

Action-space tests compare encoding and decoding by both trajectory action
spaces with float64 outputs recorded from the training implementation
(`tests/data/course_action_space_reference.safetensors`). The float64 comparison
uses a 1e-9 tolerance: float64 solves agree across LAPACK builds to about 1e-12.
The float32 comparison uses bounds of 5e-3 normalized action units, 0.5 mm, and
0.1 mrad, because float32 solves of the smoothing systems differ from float64
by up to about 1e-3 action units, and by different amounts on different LAPACK
kernels. To also check a local checkpoint's saved action-space parameters and its
driving prompt, point `ALPAMAYO2_EDGE_CHECKPOINT` at its model directory:

```bash
ALPAMAYO2_EDGE_CHECKPOINT=/path/to/model uv run --locked pytest tests/test_config.py tests/test_inputs.py
```
