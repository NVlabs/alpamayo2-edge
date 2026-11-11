<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference acceleration

The `alpamayo2_edge.accel` package provides opt-in BF16 speedups for the prompt
prefill, CoT decoding and the trajectory expert. They keep the model's weights,
BF16 precision and sampling semantics, and change how the GPU executes the same
computation. Acceleration is off by default: without the flags below, inference
runs exactly as before. Nothing is quantized. The vision tower and projector
always run eagerly.

| Option | CLI | Python | Effect on outputs |
| --- | --- | --- | --- |
| `torch.compile` + CUDA graphs | `--accel compile` or `--accel tuned` | `enable_acceleration(model, AccelConfig(...))` | Same model; fused kernels change BF16 rounding |
| Prefill attention | `--accel-attention auto\|fa3\|sdpa` | `AccelConfig(attention=...)` | Kernel choice inside the prefill graph |

## Requirements

- A CUDA GPU and a BF16 model (`--dtype bfloat16`, the default).
- FlashAttention 2 for all three graphs. The `flash` extra builds
  `flash-attn==2.8.3` from source; for the locked PyTorch 2.8 / CUDA 12
  environment on x86_64, the prebuilt release wheel installs in seconds. Pin it
  by hash:

  ```bash
  echo "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --hash=sha256:f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb" > flash-attn.txt
  uv pip install --no-deps --require-hashes -r flash-attn.txt
  ```

  The wheel needs glibc 2.32 or newer (for example Ubuntu 22.04); on older
  systems, build the `flash` extra from source instead. `uv sync` removes
  packages that are not in `uv.lock`, including this wheel: add `--inexact` to
  later syncs (for example `uv sync --locked --inexact --extra dataset --dev`),
  or reinstall the wheel afterwards.
- On Blackwell-class GPUs such as the Jetson AGX Thor Developer Kit (sm_110),
  FlashAttention 4 (`flash-attn-4`, imported as `flash_attn.cute`) replaces
  FlashAttention 2 for the decode and expert graphs; `report.json` then lists
  their kernel as `fa4`. FlashAttention 2 is preferred when both are installed.
  The [`thor-devkit/`](../thor-devkit/README.md) project locks such an environment.
- Without an importable FlashAttention 2 or 4, nothing is accelerated: the prefill and
  expert graphs use the decode part's static KV cache. `--accel` then stops with
  an error before loading the model. In Python, `enable_acceleration` emits a
  `RuntimeWarning`, and `handle.skipped` lists the parts that stay eager and why.
- FlashAttention 3 is optional and Hopper-only (`flash_attn_interface`, built
  from the FlashAttention `hopper/` directory). With `--accel-attention auto` it
  is used for the prompt prefill on sm_90 when importable. Decode and expert
  attention keep FlashAttention 2 (or 4).

## Usage

```bash
python -m alpamayo2_edge.inference \
    --model-id "$EDGE_MODEL" \
    --manifest examples/nav_samples.json \
    --accel tuned \
    --output outputs/accelerated
```

`--accel tuned` compiles the decode step with inductor's
`max-autotune-no-cudagraphs` mode and every other part with the default mode;
`--accel compile` uses the default mode everywhere and compiles faster. Before the
timed samples, the CLI runs the first input `--accel-warmup` times (default 1) to
compile and capture the graphs; outputs do not depend on the warm-up because every
sample is seeded. `report.json` records the configuration, the resolved
attention kernels, every graph build with its time, and eager fallback counts.

In Python, enable acceleration once after loading:

```python
from alpamayo2_edge.accel import AccelConfig, enable_acceleration
from alpamayo2_edge.checkpoint import load_model

model, _ = load_model("nvidia/Alpamayo2-Edge", device="cuda")
accel = enable_acceleration(model, AccelConfig(mode="tuned"))
accel.warmup(inputs)  # optional: build the graphs for one prepared input
# ... model.sample_trajectories_from_data(...) as before ...
print(accel.describe())  # parts, kernels, graph builds, fallbacks
accel.remove()  # restore the original methods
```

## What is accelerated

- **Prompt prefill**: after the eager vision encoding, the decoder stack over
  the prompt, padded to a 256-token bucket, writes its keys and values directly
  into a static KV cache.
- **CoT decoding**: every token is one replay of [decoder step, `generate`'s own
  logits processors, sampling, stopping bookkeeping], called through
  `generate(custom_generate=...)`. CUDA-graph RNG is Philox-offset based, so a
  replay draws the same random numbers as eager execution at the same state.
- **Trajectory expert**: all flow-matching steps (ten by default) form one graph
  that attends the cached prompt without copying it.

Calls outside this envelope run the original eager code, counted under
`fallbacks` in `describe()` / `report.json`:

| Situation | Graphed parts |
| --- | --- |
| One CoT (the CLI default, `1 x 1`) | All three |
| Several CoTs (`num_traj_sets * num_traj_samples > 1`) | Decode. The prefill always runs eagerly into the static cache. The expert graph replays only when all CoTs have the same length; otherwise the expert runs eagerly |
| Left-padded batches, video inputs and text-only inputs | None |
| Other logits processors, stopping criteria or streamers | All three, but sampling runs in HF's eager loop around the graphed decoder step |

## Compilation, shapes and memory

Each new input shape compiles and captures its graphs inline, so the first call
with that shape is slow. Inductor and Triton cache compiled kernels under `/tmp`
by default; set `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a persistent
directory to reuse them across processes and machines with the same GPU.

Graphs have static shapes. A new image resolution or camera set, a prompt that
crosses a 256-token bucket, a different number of CoTs, a different
`max_generation_length`, different sampling settings, or a different number of
diffusion steps builds new graphs on first use. With several CoTs, the expert
graph replays only when all CoTs have the same length. It is captured on the
first call with that shape, normally the warm-up, even though that call's CoTs
usually differ in length and run the expert eagerly. A later call whose CoTs
happen to end together therefore replays the graph instead of compiling
mid-stream.

Graphs, their static input buffers and each graph's private memory pool stay
allocated while acceleration is installed. The static KV cache, one row per CoT
sized to the prompt, `max_generation_length` and the action tokens (rounded up
to 256 tokens), persists between calls.

Accelerated outputs are not bitwise identical to eager outputs. Compiled kernels
fuse operations and change BF16 rounding, which can flip the sampled token where
two candidates are nearly tied; the CoT then continues differently.

Compiled kernels are selected when they are compiled, partly by benchmarking
candidates, and `--accel tuned` also autotunes the decode step. Two processes can
therefore select different kernels, for example when graphs are built in a
different order, and produce outputs that differ in the last bits. Reusing the
compiler cache reproduces the selection.
