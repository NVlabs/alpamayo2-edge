<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Alpamayo 2 Edge on ExecuTorch with TensorRT

[ExecuTorch](https://github.com/pytorch/executorch) is PyTorch's solution for deploying
models on devices. This example exports Alpamayo 2 Edge to ExecuTorch programs in which
[Torch-TensorRT](https://github.com/pytorch/TensorRT) engines run every part of the model
on the GPU, and runs them with the ExecuTorch runtime.

> [!IMPORTANT]
> This path needs Torch-TensorRT and ExecuTorch changes that are not released yet.
> [Unreleased changes](#unreleased-changes) lists them and marks the ones that are not
> needed. With the released packages, the KV cache writes in `prefill` fall back to
> PyTorch, the method splits into many pieces, and `export.py` stops with
> `prefill: expected one TensorRT engine`.

```
camera patches --> vision_<H>x<W> --> image embeddings --+
prompt tokens -------------------------------------------+--> prefill (whole prompt, one call)
                                                                |
                                  KV cache (shared, GPU) <------+--> decode, one token per call,
                                         |                           until the trajectory anchor token
noise --> expert_step x 10 <-------------+
              |
              v
       action --> 64 future poses
```

## How the export works

- `vision.pte` has one method per camera patch grid. Each runs the vision tower and
  projector for all images of that grid in BF16, with the layer norms and the vision
  tower's GELU in FP32. TensorRT computes those in their input type, while PyTorch
  computes them in FP32 inside. In plain BF16 that moved the predicted path several times
  more than PyTorch's own noise; with them in FP32 the path stays as close as running
  everything in FP32, at half the vision time. FP16 is not used: TensorRT's FP16 matrix
  multiplications move the path too far, even with these ops in FP32.
- `text.pte` has three methods: `prefill` (the whole prompt in one call, any length that
  fits the cache), `decode` (one reasoning token) and `expert_step` (one flow-matching
  step). They share one KV cache of a fixed size in GPU memory, and each TensorRT engine
  writes its new keys and values into it in place, with no copy.
- `prefill` and `decode` sample the next token inside the engine and return its id. They
  block trajectory tokens and the end token, like the PyTorch sampler. They apply only a
  temperature: no top-k or top-p. A temperature of 0 is greedy.
- `decode` and `expert_step` have fixed shapes, so they can replay as CUDA graphs.
- The prompt preparation (chat template, image patches, trajectory tokens, mRoPE
  positions) and the conversion of the action into poses run on the CPU with this
  package. They do not use the weights, and the timings do not include them.

The vision methods are exported for the patch grids that the 22 frames of the sample give:
14x26 for 12 images and 20x36 for 10. Frames are resized into a fixed range of pixels, so
16:9 frames such as 540x960, 720x1280 and the 1080x1920 frames of PhysicalAI-AV give the
same grids, and so do some other sizes, such as 540x1000. Frames that produce other grids,
such as 270x480, need another export, and the runner stops with an error that names the
grids. `export.py` takes its frames from `synthetic_sample()` in `host.py`, which builds
540x960 frames; to export for other frames, pass their `height` and `width` in that call.
`run.py` without `--manifest` also builds 540x960 frames.

## Setup

This package pins PyTorch 2.8, and Torch-TensorRT needs a newer one, so use a separate
environment. Install Torch-TensorRT with ExecuTorch, then this package without its
dependencies. From the repository root:

```bash
uv venv --python 3.12 .work/venv-executorch
source .work/venv-executorch/bin/activate
uv pip install "torch_tensorrt[executorch]" torchvision
uv pip install --no-deps -e .
uv pip install -r examples/executorch_tensorrt/requirements.txt
```

`uv pip check` then reports that this package asks for PyTorch 2.8 and torchvision 0.23.
That is expected in this environment.

Download the model as in [Download the model for local use](../../README.md#download-the-model-for-local-use),
so `$EDGE_MODEL` points at the model directory.

## Export

```bash
python examples/executorch_tensorrt/export.py --model-id "$EDGE_MODEL"
```

This writes `outputs/et_trt/vision.pte` and `outputs/et_trt/text.pte` (`--output-dir`
changes the folder). TensorRT builds each engine for
the GPU and TensorRT version it runs on, so export on the device that runs the programs.
`--max-seq-len` (default 4096) sets the KV cache length: prompt, reasoning and 64 action
tokens.

## Run

```bash
python examples/executorch_tensorrt/run.py \
    --model-id "$EDGE_MODEL" --cuda-graphs --runs 10 --check
```

The runner uses a synthetic driving sample, so no dataset access is needed. It prints the
time of each stage, the median over the warm runs, the reasoning and the end point of the
predicted path. Other options:

- `--check` loads the PyTorch model, runs it on the same sample with greedy reasoning
  (`top_k=1`) and the same action noise, and fails unless the reasoning tokens are equal
  and the path is finite and within 25 cm. The bound is wide because BF16 rounding alone
  moves the path. On 64 synthetic inputs (8 scenes, 4 navigation texts, a straight and a
  turning history), the PyTorch model against itself with only the attention kernel
  changed differs by 3.5 cm in the median and up to 20 cm. The exported programs differ
  from it by 3.0 cm in the median and up to 9.2 cm, with equal tokens on all 64. So the
  check catches gross errors only: an action-token position off by one can still pass.
  On these inputs the reasoning has 6 tokens, so the token comparison says little about
  `decode`.
- `--manifest examples/nav_samples.json` runs the public PhysicalAI-AV example instead.
  It needs `uv pip install physical-ai-av==0.2.0` and Hugging Face access, as in
  [Run a PhysicalAI-AV navigation example](../../README.md#run-a-physicalai-av-navigation-example).
- `--temperature` samples the reasoning; the default 0 is greedy.

## Results

Synthetic sample, greedy reasoning, median of the warm runs, in milliseconds, with nothing
else running on the GPU. The first four rows are the time of each stage; decode covers
the 6 reasoning tokens of this sample. The Total and PyTorch rows start at the vision
encoder and end after the last expert step; prompt preparation before it and the
conversion to poses after it are not included. In between, PyTorch also computes the mRoPE
positions, and the ExecuTorch runner copies each method's inputs and outputs between CPU
and GPU. The ExecuTorch rows use CUDA graphs on `decode` and `expert_step`. The PyTorch
rows run this package's model with its default path and with `--accel tuned`, with
`top_k=1`, on the same PyTorch 2.15 (with flash-attn 2.8.3 for `--accel tuned`).

| Stage | A100 | DGX Spark | Jetson AGX Thor |
| --- | --- | --- | --- |
| Vision | 96 | 213 | 147 |
| Prefill | 63 | 159 | 115 |
| Decode | 51 | 110 | 104 |
| Expert (10 steps) | 64 | 128 | 129 |
| **Total** | **273** | **611** | **495** |
| PyTorch, default | 821 | 1038 | 1067 |
| PyTorch, `--accel tuned` | 217 | 617 | 590 |

`--check`: on all three devices the reasoning tokens match the PyTorch model. The
largest path error is 0.7 to 1.5 cm, depending on the build, since TensorRT picks its
kernels again on every build.

What was tested: on an A100 with x86_64 Linux, the setup commands above, `export.py`,
`run.py --check` with and without `--cuda-graphs`, the 64-input comparison described under
`--check`, and `export.py` with only the released packages, which stops at `prefill` as
described at the top. On a DGX Spark and a Jetson AGX Thor, `export.py` and `run.py --check`
with and without `--cuda-graphs`, with Torch-TensorRT and ExecuTorch built there with the
same changes.

## Unreleased changes

The results above used the released Torch-TensorRT 2.15.0 with the Python changes of the
series below applied, and the ExecuTorch delegate built from the C++ changes below. They
come from pre-release revisions of these changes, and will be refreshed when the changes
land.

Torch-TensorRT:

- **fix(dynamo): write copy-back buffers when the compiled module runs in PyTorch.**
- **fix(lowering): update a cache in place only when nothing else reads the old cache.**
  Decides when an engine may write a buffer in place without another layer reading the
  new values too early. Every cache write here goes through it.
- **fix(conversion): do not update a KV cache in place where TensorRT cannot.**
- **fix(conversion): keep integer caches off the KV cache layer.**
- **fix(partitioning): keep an in-place buffer write in an engine at any size.**
- **feat(conversion): write multi-token KV-cache updates from index_copy in place.**
  Lets `prefill` write the whole prompt and `expert_step` write 64 tokens into the cache
  in place, with a dynamic prompt length in one engine.
- **feat(conversion): convert index_copy with a dynamic source length.** Not needed here:
  a run without it gave the same result.

The ExecuTorch delegate of Torch-TensorRT (`torch_tensorrt_executorch_runtime`), built
from main with these changes and against the ExecuTorch below, because the released
package requires the released ExecuTorch:

- **fix(executorch): copy only the moving bindings through CUDA graph buffers.**
- **perf(executorch): replay CUDA graphs for engines that update a cache in place.**
  Builds on the change above. Together they let `--cuda-graphs` replay `decode` and
  `expert_step`. Without them, run without `--cuda-graphs`: the results are the same,
  only slower.
- **feat(executorch): accept TensorRT engines with shape tensor inputs**, with the changes
  it builds on. Not needed here: a delegate without it ran these programs with the same
  result.

ExecuTorch main, plus these changes so that the three text methods share one cache in
GPU memory:

- **Record in the program which planned buffers its methods share.**
- **Give every method one allocation of each buffer the program marks shared.**
- **Plan mutable buffers shared between methods in device memory.**

## Limitations

- The runner is in Python. A C++ runner would also need the prompt preparation in C++.
- An exported program fits the patch grids of the frames it was exported with.
- The reasoning sampler applies a temperature only. The PyTorch sampler uses top-p 0.98
  at temperature 0.6 by default. Greedy reasoning is the same in both.
- The programs hold some weights twice. `prefill` and `decode` each hold the language
  model weights, so `text.pte` is 8.3 GB, and each vision method holds the vision
  weights, so `vision.pte` is 2.0 GB, against a 5.3 GB checkpoint. The runner uses
  about 12 GB of GPU memory on the A100, so it does not fit the 8 GB of a Jetson Orin
  Nano. Sharing weights between engines needs support in Torch-TensorRT.
- Greedy reasoning matches the PyTorch model when its best next token is clear. When the
  two best tokens are closer than BF16 rounding, either implementation can pick either
  one. For example, with the navigation text "Continue straight." repeated 150 times (a
  3705-token prompt), the PyTorch model run fully in FP32 scores the first tokens "E" and
  "B" only 0.013 apart, less than one BF16 step at that size. Which one an export picks
  depends on the kernels TensorRT chooses when it builds the engines. After "B" the model
  does not reach the trajectory within 256 tokens, also fully in FP32, and the runner
  stops with an error.
