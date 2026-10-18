<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Streaming prefill simulation

`alpamayo2_edge.streaming` reproduces in PyTorch what a streaming runtime with a
per-image K/V cache computes, so that its effect on reasoning and trajectories
can be measured, and trained against, without that runtime.

## What a streaming runtime computes

Input arrives at 10 Hz. Each prompt holds 22 images: five frames of each front
camera (-2, -1, -0.5, -0.1 and 0 s) and three of each other camera (-1, -0.1 and
0 s). Only the newest frame of each camera is new at a tick; every other frame
was the newest frame of an earlier tick. With the EA checkpoint's prompt, the text
and the six new images are about a third of the prompt rows.

A streaming runtime caches each image's K/V when the image is new and, at later
ticks, splices it into the KV cache instead of recomputing it. It recomputes the
text and the new images, which attend to the spliced rows. Three properties
follow:

- **Stale content.** An older image keeps the K/V it got at the tick where it was
  new, computed on that tick's context. That context was itself partly spliced, so
  the staleness compounds from tick to tick.
- **Exact positions.** The time labels and the camera-major layout move every image
  to a new slot at each tick. The runtime rotates a reused key to the new mRoPE
  position, so only the content differs from a full prefill, not the positions.
- **Cold start.** The first ticks of a stream find older frames missing from the
  cache and recompute them. After two seconds, the cache holds every frame a
  prompt can contain, and only the newest frames are recomputed.

Ordinary prefix caching does not apply: the older images are interleaved with
text that changes every tick, so they never form a prefix of the prompt.

## How the simulation reproduces it

`StreamingPrefill` runs the whole prompt, with forward hooks on every
language-model layer:

- the output of `k_norm` (K before RoPE) and of `v_proj` (V) are replaced, in the
  rows of every reused image, by that image's cached rows. The layer then applies
  RoPE at the current positions, which is the runtime's rotation to the new slot;
- the same rows of every recomputed image are captured into the cache, keyed by
  `image_frame_ids` (stream, camera, capture time);
- at the start of each tick, frames captured before the oldest image of its prompt
  are evicted; prompts only move forward in time, so those frames cannot appear
  again. This needs no time margin for decoded capture times that precede the
  requested ones.

Tokens interact only through attention K and V. Once an image's K/V rows are
replaced in every layer, its own hidden states reach no other token, so running
the whole prompt and overwriting those rows is the same function as computing
only the recomputed rows. The simulation therefore matches the runtime's
semantics but not its cost: every image still runs through the vision tower and
every row through the language model. DeepStack features of reused images are
added to rows whose output is discarded, as in a runtime that skips them.

The hooks act only on forwards that start a sequence and cover every image (the
prompt prefill), so CoT decoding and the expert are unchanged and run on the
spliced cache.

## Python API

```python
from alpamayo2_edge.helper import prepare_model_inputs, to_device
from alpamayo2_edge.streaming import StreamingPrefill

stream = StreamingPrefill(model)  # one per stream
for sample in samples[:-1]:  # consecutive 0.1 s ticks, oldest first
    prepared = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), "cuda")
    plan = stream.advance(sample, prepared)  # prefill only; returns what was reused

sample = samples[-1]
prepared = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), "cuda")
with stream.tick(sample, prepared) as plan:
    xyz, rot, _, extra = model.sample_trajectories_from_data(prepared, return_extra=True)
print(plan.summary())  # prompt, reused and recomputed tokens; frames missing from the cache
```

- `tick(..., reuse=False)` recomputes every image, a full prefill, and still
  captures its K/V. Comparing the two caches with `kv_difference` gives cosine,
  relative L1 and relative L2 per layer for K and V.
- `stream_ticks_us(t0_us, warmup_us)` lists the tick times of a stream that starts
  cold `warmup_us` before `t0_us`. A warm-up of at least two seconds fills the
  cache by `t0`. The cold-start tick needs its own two-second camera history, so
  the warm-up can be at most `t0` minus two seconds.
- Samples need `camera_indices` and `absolute_timestamps`, as
  `load_physical_aiavdataset` returns, and a `clip_id` or an explicit
  `StreamingPrefill(model, stream=...)` name.

## Compare a streamed and a full prefill

```bash
python -m alpamayo2_edge.streaming \
    --model-id "$EDGE_MODEL" \
    --manifest examples/nav_samples.json \
    --warmup-s 2.0 \
    --output outputs/streaming
```

The example's `t0` is 4.0 seconds, so a two-second warm-up is the longest it
allows; for a later `t0`, `--warmup-s 3.0` (the default) is also possible.

The command loads every 0.1 s tick from the cold start to `t0` through
PhysicalAI-AV, replays the stream, then runs inference at `t0` twice with the same
seed: on the streamed prefix and on a full prefill. CoT decoding is greedy by
default (`--top-k 1`), so the two runs differ only by their prefix.
`streaming_report.json` contains each tick's reuse counts, both CoTs, the
distance between the two predicted trajectories, and the K/V difference to the
full prefill for the reused and the newest images. A difference between the two
trajectories measures the streamed prefix against the same model with a full
prefill; it is not a comparison with the ground-truth future.

## Limitations

- Run the model eagerly. The hooks do not reach into the compiled graphs of
  `alpamayo2_edge.accel`; `VisionFeatureCache` can be combined with it to skip
  re-encoding older frames.
- One prompt per tick: the prefill runs at batch size 1, as
  `sample_trajectories_from_data` does.
- Frames are matched by camera and capture time. If a camera repeats a frame (the
  decoder returns the latest frame at or before each request), the repeated frame
  is reused like any other cached frame.

The CPU tests in `tests/test_streaming.py` establish, on a small Qwen3-VL with
DeepStack: a cold tick equals plain inference bit for bit; splicing a tick's own
K/V into itself is exact; the pixels of reused images cannot change any other
row; cached keys are captured before RoPE; and reused images' K/V match a full
prefill in the first layer but not in deeper ones, as stale context implies.
