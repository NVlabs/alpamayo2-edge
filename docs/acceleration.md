<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference acceleration

The `alpamayo2_edge.accel` package provides opt-in BF16 speedups. They keep the
model's weights, BF16 precision and sampling semantics, and change how the GPU
executes the same computation. Every option is off by default: without the flags
below, inference runs exactly as before. No option quantizes weights or
activations.

| Option | CLI | Python | Effect on outputs |
| --- | --- | --- | --- |
| `torch.compile` + CUDA graphs | `--accel compile` or `--accel tuned` | `enable_acceleration(model, AccelConfig(...))` | Same model; fused kernels change BF16 rounding ([parity](validation.md#parity-on-h100)) |
| Attention selection | `--accel-attention auto\|fa3\|sdpa` | `AccelConfig(attention=...)` | Kernel choice inside the graphs |
| ViT feature cache | Python only | `VisionFeatureCache(model)` | Reuses identical per-frame features; needs consecutive frames |
| GEMM patch embedding | `--patch-embed linear` | `use_linear_patch_embedding(model)` | Same math as the Conv3d, about 1 BF16 ulp apart |

## Requirements

- A CUDA GPU and a BF16 model (`--dtype bfloat16`, the default).
- FlashAttention 2 for the prefill, decode and expert graphs. The `flash` extra
  builds `flash-attn==2.8.3` from source; for the locked PyTorch 2.8 / CUDA 12
  environment on x86_64, a prebuilt wheel installs in seconds:

  ```bash
  uv pip install --no-deps \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  ```

  The wheel needs glibc 2.32 or newer (for example Ubuntu 22.04); on older
  systems, build the `flash` extra from source instead. Without an importable
  `flash_attn`, only the vision tower is accelerated; `report.json` lists the
  skipped parts and the reason.
- FlashAttention 3 is optional and Hopper-only (`flash_attn_interface`, built
  from the FlashAttention `hopper/` directory). With `--accel-attention auto` it
  is used for the vision tower and prompt prefill on sm_90 when importable.
  Decode and expert attention keep FlashAttention 2, which is faster for these
  short queries.

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

model, _ = load_model("nvidia/EA-Alpamayo2-Edge", device="cuda")
accel = enable_acceleration(model, AccelConfig(mode="tuned"))
accel.warmup(inputs)  # optional: build the graphs for one prepared input
# ... model.sample_trajectories_from_data(...) as before ...
print(accel.describe())  # parts, kernels, graph builds, fallbacks
accel.remove()           # restore the original methods
```

## What is accelerated

- **Vision tower**: images that share a patch grid (side and front cameras differ)
  run as one graph replay with batched attention, and rows are interleaved back
  in image order. Qwen3-VL encodes images independently, so grouping does not
  change the computation.
- **Prompt prefill**: the decoder stack over the prompt, padded to a 256-token
  bucket, writes its keys and values directly into a static KV cache.
- **CoT decoding**: every token is one replay of [decoder step, `generate`'s own
  logits processors, sampling, stopping bookkeeping], called through
  `generate(custom_generate=...)`. CUDA-graph RNG is Philox-offset based, so a
  replay draws the same random numbers as eager execution at the same state.
- **Trajectory expert**: all ten flow-matching steps form one graph that attends
  the cached prompt without copying it.

Calls outside this envelope run the original eager code, counted under
`fallbacks` in `describe()` / `report.json`:

| Situation | Graphed parts |
| --- | --- |
| One CoT (the CLI default, `1 x 1`) | All four |
| Several CoTs (`num_traj_sets * num_traj_samples > 1`) | Vision and decode; prefill runs eagerly into the static cache; the expert is graphed only when all CoTs have the same length |
| Left-padded batches, text-only or video inputs | None (eager) |
| Other logits processors, stopping criteria or streamers | Decode step only, inside HF's sampling loop |

## Compilation, shapes and memory

Each new input shape compiles and captures its graphs inline, so the first call
is slow. On an H100, the eight graphs for one and six CoTs took 123 s to build
with `--accel compile` and 170 s with `--accel tuned` from an empty compiler
cache, and the first one-CoT call alone took about 110 s. With a populated cache,
building six of these graphs took 23 s. Inductor and Triton cache compiled
kernels under `/tmp` by default; set `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` to a persistent directory to reuse them across processes and
machines with the same GPU.

Graphs have static shapes. A new image resolution or camera set, a prompt that
crosses a 256-token bucket, a different number of CoTs, a different
`max_generation_length`, different sampling settings, or a different number of
diffusion steps builds new graphs on first use. Graphs and their static buffers
stay allocated for the process lifetime. On the H100, peak allocated memory grew
from 5.6 to 5.9 GB for one CoT and from 8.6 to 8.8 GB for six CoTs.

Accelerated outputs are not bitwise identical to eager outputs. Compiled kernels
fuse operations and change BF16 rounding, which can flip the sampled token where
two candidates are nearly tied; the CoT then continues differently.

## ViT feature cache

In streaming use, consecutive calls share most frames: a frame that is current at
one step reappears at its past offsets (-0.1 s, -0.5 s, -1 s and -2 s) in later
prompts. Qwen3-VL encodes every image independently, so `VisionFeatureCache`
stores the vision features of each frame and encodes only frames it has not seen,
normally the newest frame of each camera:

```python
import torch

from alpamayo2_edge.accel import VisionFeatureCache, image_frame_ids
from alpamayo2_edge.helper import prepare_model_inputs, to_device

cache = VisionFeatureCache(model)  # after enable_acceleration, if used
for sample in stream:  # consecutive time steps of one drive
    prepared = prepare_model_inputs(sample, model.config, model.tokenizer)
    prepared = cache.prepare(prepared, image_frame_ids(sample))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, rot, _ = model.sample_trajectories_from_data(to_device(prepared, "cuda"))
cache.clear()  # at the end of a stream
```

- `prepare` must run on the inputs of every call; it removes the cached frames'
  patches, so only new frames are uploaded and encoded.
- A frame is identified by `(camera, capture time)`, its patch grid and a
  fingerprint of sampled pixel rows. Pass `full_digest=True` to hash every pixel
  (about 2 ms of CPU time per image).
- It only helps when calls see the same frames again: each camera's frames must
  stay in the cache from capture until their oldest offset (2 s for the front
  cameras). `max_images` (default 256, at most about 0.75 GB) bounds the cache
  with least-recently-used eviction.
- With eager attention, cached features matched uncached encoding bit for bit
  in our checks; kernels whose rounding depends on how many images are encoded
  together can differ in the last bit. With the compiled vision graphs, a frame
  encoded in a smaller batch can differ by one or two BF16 ulps.
- On an H100 in streaming steady state, encoding the six newest frames instead
  of all 22 saved 33 ms per call in eager mode and 13 ms with the compiled
  graphs.

## GEMM patch embedding

The vision patch embedding is a Conv3d whose kernel equals its stride, which is
one dot product per patch. `--patch-embed linear` computes it as a single GEMM.
It is required for usable vision latency where BF16 Conv3d falls back to a slow
kernel. With PyTorch 2.9 on DGX Spark (GB10, sm_121), an earlier prototype
measured about 35 s for the 22-image vision call with the Conv3d. On an H100,
where the Conv3d is fast, the GEMM made no measurable difference. The two agree
to rounding, about one BF16 ulp per output.
