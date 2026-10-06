<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Alpamayo 2 Edge

Standalone PyTorch inference for the Qwen3-VL 2B Alpamayo 2 Edge checkpoint and its
trajectory diffusion expert. Given six-camera observations, ego history, and a
navigation instruction, the model generates driving reasoning and 64 future poses
at 0.1-second intervals.

This is a **pre-release implementation**. The example below uses the private
[Internal-Alpamayo2-Edge checkpoint bundle](https://huggingface.co/nvidia/Internal-Alpamayo2-Edge/tree/7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4/2b/1002_model1_v3_plain_vqa1000_step6530),
which requires authorized Hugging Face access. A public model URL has not been
selected. This implementation currently supports the Qwen3-VL checkpoint and its
22-image, 20-pose input profile.

## Setup

Use Python 3.12 and a CUDA GPU for full model inference. Dependencies pin PyTorch
2.8.0, torchvision 0.23.0, and Transformers 5.17.0. From this repository:

```bash
uv sync --locked --dev
source .venv/bin/activate
```

The default attention backend is PyTorch SDPA; installation does not build
FlashAttention. FlashAttention 2.8.3 is an optional `flash` extra for environments
with compatible CUDA build prerequisites. To select it after installation, pass
`--attn-implementation flash_attention_2`. The bundle's saved references were
produced with FA2, BF16, and an H100; a different backend or runtime can change
sampled text and trajectories.

## Download the checkpoint and examples

Authenticate using `hf auth login` or an existing `HF_TOKEN`, then download the
immutable revision. The bundle contains both model shards, tokenizer and processor
assets, ten raw examples, and reference outputs; allow approximately 8 GB of disk
space.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/Internal-Alpamayo2-Edge",
    revision="7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4",
    allow_patterns=["2b/1002_model1_v3_plain_vqa1000_step6530/**"],
    local_dir=".work/checkpoint",
)
PY

export EDGE_BUNDLE="$PWD/.work/checkpoint/2b/1002_model1_v3_plain_vqa1000_step6530"
(cd "$EDGE_BUNDLE" && sha256sum --check checksums.sha256)
```

The model is under **`$EDGE_BUNDLE/model`**, not the bundle root. The loader accepts
either directory and resolves all tokenizer and processor assets locally. Legacy
checkpoint class names are normalized in memory; checkpoint files remain intact.

## Run inference

Run the first bundled example:

```bash
python -m alpamayo2_edge.inference \
    --bundle "$EDGE_BUNDLE" \
    --output outputs/smoke
```

Run all ten examples, or select individual IDs with repeatable
`--sample-id sample_003` arguments:

```bash
python -m alpamayo2_edge.inference \
    --bundle "$EDGE_BUNDLE" \
    --all-samples \
    --output outputs/all-samples
```

Bundle inference reads sampling settings from `manifest.json`: seed 43,
`num_traj_sets=6`, `num_traj_samples=1`, temperature 0.6, top-p 0.98, no top-k limit,
at most 256 generated tokens, and ten Euler diffusion steps. Each set has one
independently sampled reasoning continuation; each sample within that set has an
expert trajectory conditioned on that continuation. Thus `6 / 1` generates six
reasoning continuations and six total trajectories, while `1 / 6` generates one
reasoning continuation and six trajectories. Override these settings explicitly:

```bash
python -m alpamayo2_edge.inference \
    --bundle "$EDGE_BUNDLE" \
    --num-traj-sets 1 --num-traj-samples 6 \
    --output outputs/one-reasoning-six-trajectories
```

Use a standalone local model directory with your own raw sample directory:

```bash
python -m alpamayo2_edge.inference \
    --model-id "$EDGE_BUNDLE/model" \
    --sample "$EDGE_BUNDLE/samples/sample_000" \
    --num-traj-sets 6 --num-traj-samples 1 \
    --output outputs/local-sample
```

`--model-id` also accepts a Hugging Face repository ID, with optional `--revision`
and `--subfolder` arguments. For the private bundle above, its model subfolder is
`2b/1002_model1_v3_plain_vqa1000_step6530/model`. Standalone model mode defaults to
one reasoning continuation and one trajectory because it does not read a bundle
manifest. Other options include `--seed`, `--diffusion-steps`,
`--max-generation-length`, `--device`, and `--dtype`; see `--help`.

Each sample's output directory contains `output.safetensors` with `pred_xyz`,
`pred_rot`, and generated token IDs; `output.json` with decoded text and shape
checks; `packed.safetensors` with the actual model inputs; and `prompt.txt` with the
fused prompt. `report.json` in the output root records runtime versions, attention
backend, sampling settings, and checkpoint loading diagnostics. With the bundle's
default settings, positions have shape `[1, 6, 1, 64, 3]` and rotations have shape
`[1, 6, 1, 64, 3, 3]`.

## Raw sample contract

A sample directory contains `sample.json` and `tensors.safetensors`
(`input.safetensors` is also accepted). Store these five tensors:

| Field | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `image_frames` | `uint8` | `[22, 1, 3, H, W]` | Contiguous CPU RGB images; bundled images are 1080 × 1920. |
| `camera_indices` | `int64` | `[22]` | Camera ID for each image. |
| `absolute_timestamps` | `int64` | `[22]` or `[22, 1]` | Captured image timestamps in microseconds. |
| `ego_history_xyz` | Floating point | `[1, 20, 3]` | Twenty finite ego positions. |
| `ego_history_rot` | Floating point | `[1, 20, 3, 3]` | Corresponding finite rotation matrices. |

History poses span −1.9 through 0 seconds at 0.1-second intervals and produce 57
delta tokens; they must already use the checkpoint's ego-history coordinate
convention. Adding a leading batch dimension to either history tensor is accepted.
Inference requires exactly the following camera counts and sampling profile:

| Cameras | Images per camera | Intended image offsets (seconds) | Pixel budget (min–max) |
| --- | --- | --- | --- |
| 1, 6: front and front telephoto | 5 | −2, −1, −0.5, −0.1, 0 | 163840–196608 |
| 0, 2, 3, 5: remaining cameras | 3 | −1, −0.1, 0 | 81920–98304 |

Actual capture timestamps may have sensor jitter. Preprocessing sorts images by
camera and timestamp together, renders relative-time labels, resizes in `uint8`
with bicubic interpolation, and applies the checkpoint's normalization. Frame
ordering, timestamps, and per-camera pixel budgets are part of the model input.

`sample.json` must include nonempty navigation text as a one-element list, for
example `{"nav_text": ["Continue straight"]}`. The optional
`camera_pixel_bounds` object maps camera IDs to `[min_pixels, max_pixels]`; the
table's defaults apply when omitted. Bundled metadata also records
`history_num_steps=20` and source provenance. Future trajectory labels are stored
separately and are not needed for inference.

## Python API

```python
import torch

from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device
from alpamayo2_edge.sample import load_sample

model, loading_info = load_model("/path/to/bundle/model", device="cuda")
sample = load_sample("/path/to/sample_000")
inputs = to_device(
    prepare_model_inputs(sample, model.config, model.tokenizer), device="cuda"
)
torch.manual_seed(43)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    xyz, rotations, _, extra = model.sample_trajectories_from_data(
        inputs,
        num_traj_sets=6,
        num_traj_samples=1,
        max_generation_length=256,
        diffusion_kwargs={
            "inference_step": 10,
            "generator": torch.Generator(device="cuda").manual_seed(43),
        },
        return_extra=True,
    )
```

`load_model` rejects missing, unexpected, or mismatched checkpoint parameters.
The third model return value is a placeholder log probability and should not be
used to rank candidates. The CLI checks trajectory shapes and finite values.

For development, run `uv run --locked pytest`. See
[validation results and reproduction commands](docs/validation.md) for measured
checkpoint loading, preprocessing, and GPU inference results, including remaining
differences from the supplied reference outputs. See
[Hugging Face release preparation](docs/huggingface_release.md) for the exact
configuration and artifact contract. Inference components are adapted from
[Alpamayo 2 Super, commit 5e7975f](https://github.com/NVlabs/alpamayo2/tree/5e7975f4a2100ee8ac1a62b79239bbabefddcbef).
Source code is distributed under [Apache-2.0](LICENSE); checkpoint access and its
model-card terms are separate.
