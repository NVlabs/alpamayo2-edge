<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preparing an Alpamayo 2 Edge Hugging Face release

The current inference target is the private pre-release bundle
[`nvidia/Internal-Alpamayo2-Edge`](https://huggingface.co/nvidia/Internal-Alpamayo2-Edge/tree/7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4/2b/1002_model1_v3_plain_vqa1000_step6530),
subfolder `2b/1002_model1_v3_plain_vqa1000_step6530`, immutable revision
`7db095f3f1c54e2236f96e1baf12c2bc3c5ee9e4`. Its model files are one directory deeper,
under `model/`. A public repository name has not been selected.

The loader accepts this original bundle without editing it. It copies the config
in memory and translates known class/module identifiers into this package. The
same translations can be applied to a staged release's `config.json` to make the
saved configuration describe the standalone implementation. Keep the original
bundle and checksums as provenance; calculate new metadata checksums for the
staged release.

## Configuration identifiers

Use these wrapper identifiers in a release config:

| Field | Original bundle | Standalone release |
| --- | --- | --- |
| `model_type` | `alpamayo3` | `alpamayo2_edge` |
| `architectures` | `["Alpamayo3"]` | `["Alpamayo2Edge"]` |
| `expert_config.model_type` | `alpamayo1_expert` | `alpamayo2_edge_expert` |

`Alpamayo3` changed the training compute schedule while preserving the VLM and
expert parameter structure. Its tensors load under the existing `vlm.*` and
`expert.*` keys. The standalone inference wrapper does not require conversion of
trained tensors.

Replace only the following known Hydra `_target_` values wherever they occur:

| Original target | Standalone target |
| --- | --- |
| `alpamayo.tokenizer.trajectory.delta_tokenizer.DeltaTrajectoryTokenizer` | `alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer` |
| `alpamayo.action_space.discrete_action_space.DiscreteTrajectoryTokenizer` | `alpamayo2_edge.action_space.discrete_action_space.DiscreteTrajectoryTokenizer` |
| `alpamayo.action_space.UnicycleAccelCurvatureActionSpace` | `alpamayo2_edge.action_space.UnicycleAccelCurvatureActionSpace` |
| `alpamayo.diffusion.flow_matching.FlowMatching` | `alpamayo2_edge.diffusion.flow_matching.FlowMatching` |
| `reasoning_vla.submodules.action_in_proj.PerWaypointActionInProjV2` | `alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2` |

`torch.nn.Linear` remains unchanged. The implementation rejects unknown targets
before instantiation. Use `normalize_checkpoint_config` in
`alpamayo2_edge.config` to perform these translations on a copy.

The original `vlm_name_or_path` and `vlm_config._name_or_path` contain
`9ce19a195e423419c349abfc86fd07178b230561`. This is stale provenance, not a usable
backbone repository ID. For staged metadata, remove stale machine-specific
`_name_or_path` values and set `vlm_name_or_path` to `null`, or replace it with an
explicitly selected release repository ID once one exists. The shipped loader
sets the wrapper's `_name_or_path` to the resolved local model directory and loads
the tokenizer and processor from that directory. It never needs to fetch a
separate base model.

This release expects the Python package to be installed and uses
`alpamayo2_edge.checkpoint.load_model`. It does not require `trust_remote_code` or
an `auto_map` pointing to the internal training package.

## Preserve the trained model

Copy the two weight shards and the weight index unchanged. The original checkpoint
has 947 BF16 tensors, 2,653,202,434 parameters, and 5,306,404,868 tensor-data bytes.
Its 198,776,834-parameter expert is included in those totals. The shard files are:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `00000.safetensors` | 4270871424 | `7d6546861cfee8db3df19b69066a1723c6a184f2028cbee5e1a7041c33ef5bcb` |
| `00001.safetensors` | 1035650556 | `ac33ac9449e7bbbc83c655b459f93fd895ae7e30eabc68bfe61ec3c8028b147e` |

Keep all learned values, tensor shapes, attention head dimensions, layer counts,
RoPE parameters, trajectory quantization settings, action normalization values,
and diffusion settings. In particular:

- Keep `vlm_class="Qwen3VLForConditionalGeneration"`, VLM hidden size 2048, and
  expert hidden size 512. Both text stacks have 28 layers, 16 query heads, 8 KV
  heads, and head dimension 128; the expert's attention projection width exceeds
  its hidden size intentionally.
- Preserve `rope_parameters`, including theta 5000000, interleaved MRoPE, and
  sections `[24, 20, 20]` on both text configs.
- Keep `expert_non_causal_attention=true`. Expert future tokens attend to one
  another and to the valid VLM prefix. The SDPA implementation supplies an
  explicit noncausal mask; setting a keyword after building a causal mask does
  not provide the same behavior.
- Keep the saved tokenizer and embedding size 155776. The tokenizer has 155697
  entries; its padded embedding rows are intentional. Do not add, reorder, or
  recreate tokens, resize embeddings, or infer token IDs from a different base
  tokenizer.

The saved trajectory token IDs are:

| Field | ID |
| --- | ---: |
| `history_id0` | 151669 |
| `future_id0` | 152669 |
| `history_start` | 155674 |
| `history_end` | 155676 |
| `history_pad` | 155684 |
| `future_start` | 155681 |
| `future_end` | 155683 |
| `future_pad` | 155685 |

The history/future vocabularies have 1000/3000 entries. Discrete future-token count
128 describes the saved future tokenizer; expert inference generates 64 continuous
action waypoints. The future-start token anchors the expert after text generation.

## Publish a self-contained processor and input contract

Include `config.json`, both shards, `model.safetensors.index.json`,
`tokenizer.json`, `tokenizer_config.json`, `added_tokens.json`,
`special_tokens_map.json`, `vocab.json`, `chat_template.jinja`,
`preprocessor_config.json`, `processor_config.json`,
`video_preprocessor_config.json`, and `generation_config.json`. Keep the
checkpoint's complete processor settings even though this release uses images.
The pinned runtime is Transformers 5.17.0; the available Transformers 4.57.6
environment fails to read this saved tokenizer configuration.

Preserve these model and sample metadata values:

| Contract | Required value for this checkpoint |
| --- | --- |
| Frame presentation | `frame_label="relative_time"`, `token_layout="camera_ts"`, `include_camera_ids=true`, `encode_as_video=false` |
| Ego history | 20 positions/rotations, 0.1-second spacing, −1.9 through 0 seconds |
| History encoding | `pad_origin_at_beginning=false`, `tokens_per_history_traj=57` |
| Images | 22 raw RGB images across camera IDs 0, 1, 2, 3, 5, 6 |
| Front cameras 1 and 6 | Five frames each; pixel bounds 163840–196608 |
| Other four cameras | Three frames each; pixel bounds 81920–98304 |
| Timestamp units | Captured `int64` microseconds, retained with their corresponding images |
| Navigation | Nonempty `nav_text` one-element list |
| Prompt order | Image, trajectory history, navigation instruction, task |

The config's `min_pixels=163840` and `max_pixels=196608` describe the front-camera
budget. Retain the per-camera overrides in sample metadata or an explicit input
profile; applying the front-camera budget to all images changes the input.
Restore integer camera-map keys after JSON loading. Sort frame-aligned fields
together, resize `uint8` pixels before normalization, and preserve relative-time
labels and component newlines. Replacing these operations with a generic VLM
prompt changes checkpoint behavior.

Keep the saved `chat_template.jinja` unchanged and use the Edge conversation
recipe: images, trajectory history, navigation instruction, then task text;
the task requests reasoning followed by a future trajectory. The navigation
prefix is `Navigation input is: `, the task prefix is `Your task: `, and logical
components are separated by newlines. This plain-text recipe does not prefill
`<|cot_start|>` or use route wrapper tokens.

Relative timestamps use the last capture **after camera-major sorting** as their
zero point: the latest camera-6 frame in this input profile. Preserve this
convention even when another camera's latest capture or `event_t0_us` is later.
Subtract timestamps in `int64` microseconds, then convert the difference to
seconds and format one decimal place, normalizing negative zero to `0.0`.
Place each `<t=...s>` label immediately before its corresponding image. The
[source and input-boundary audit](validation.md#chat-template-and-timestamp-audit)
found no required template or timestamp config change for this checkpoint.

For a bundle release, retain `manifest.json`, `samples_manifest.json`, ten
`samples/sample_NNN/{sample.json,tensors.safetensors}` directories, and the
corresponding `reference/` artifacts. Ground-truth trajectories are separate
optional evaluation data and are never loaded by the inference runner. Preserve
sample hashes, image/navigation provenance, `history.num_steps=20`,
`history.time_step_seconds=0.1`, and `input_profile="camera_subsampled"`.

## Record runtime and verification evidence

The supplied references were recorded with Python 3.12, PyTorch 2.8.0+cu128,
Transformers 5.17.0, FlashAttention 2.8.3, BF16, and an H100. They use seed 43, six
reasoning continuations with one expert sample each, top-p 0.98, temperature 0.6,
no top-k limit, maximum generation length 256, and ten Euler diffusion steps.
Retain these settings in the bundle manifest. `num_traj_sets=1` and
`num_traj_samples=6` define a different workload even though both produce six
trajectories.

The original bundle's validation reports describe that original runtime. They do
not certify a new code revision or SDPA replay. For the staged release, record
the implementation commit, immutable HF revision, actual package versions,
attention backend, GPU, and sampling settings alongside new results. Check that
the entire trained state loads with no missing, unexpected, or mismatched
parameters, and that generated trajectories have the expected shapes and finite
values. Compare packed token IDs, image grids, image patches, histories, and the
fused prompt before attributing output differences to model execution.

Reference `generated_ids` contain continuation tokens through the future anchor
and use `-1` padding. Plain reasoning text has no required CoT wrapper tokens.
Positions and rotations use `[batch, num_traj_sets, num_traj_samples, 64, ...]`;
the bundle defaults produce `[1, 6, 1, 64, 3]` positions. Exact sampled-output
reproduction requires the original runtime and sampling behavior; a finite-output
smoke run alone does not establish numerical parity.

The standalone inference components originate from
[Alpamayo 2 Super commit 5e7975f4a2100ee8ac1a62b79239bbabefddcbef](https://github.com/NVlabs/alpamayo2/tree/5e7975f4a2100ee8ac1a62b79239bbabefddcbef).
Retain source copyright/license notices and provide the separately applicable
checkpoint model card and terms with any release.
