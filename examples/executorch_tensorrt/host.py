# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side steps around the exported programs, built from the checkpoint config.

Preparing the prompt (chat template, image patches, trajectory tokens, mRoPE positions)
and turning the predicted action into a trajectory need no model weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import hydra.utils
import torch
from transformers import Cosmos3EdgeForConditionalGeneration

from alpamayo2_edge.checkpoint import resolve_checkpoint
from alpamayo2_edge.config import (
    Alpamayo2EdgeConfig,
    build_alpamayo2_edge_tokenizer,
    normalize_checkpoint_config,
)
from alpamayo2_edge.helper import prepare_model_inputs
from alpamayo2_edge.models.token_utils import extract_text_tokens
from alpamayo2_edge.models.utils import fuse_traj_tokens


@dataclass
class Prompt:
    tokens: torch.Tensor  # [1, P] int64
    positions: torch.Tensor  # [3, 1, P] int64 mRoPE positions
    rope_delta: int  # text position offset for tokens after the prompt
    patches: list[torch.Tensor]  # per image, [H * W, 768] fp32
    grids: list[tuple[int, int]]  # per image, patch grid (H, W)
    history_xyz: torch.Tensor  # [1, 20, 3] ego history positions
    history_rot: torch.Tensor  # [1, 20, 3, 3] ego history rotations


class HostPipeline:
    def __init__(self, model_id: str) -> None:
        path = resolve_checkpoint(model_id)
        raw = json.loads((path / "config.json").read_text())
        config = Alpamayo2EdgeConfig(**normalize_checkpoint_config(raw))
        config._name_or_path = str(path)
        self.config = config
        self.tokenizer = build_alpamayo2_edge_tokenizer(
            str(path), config.history_vocab_size, config.future_vocab_size
        )
        self.history_tokenizer = hydra.utils.instantiate(
            config.hist_traj_tokenizer_cfg, load_weights=False
        )
        self.future_tokenizer = hydra.utils.instantiate(
            config.future_traj_tokenizer_cfg, load_weights=False
        )
        # The loaded model keeps the action statistics in its weight dtype.
        self.action_space = hydra.utils.instantiate(config.expert_config.action_space_cfg).to(
            torch.bfloat16
        )
        # Only get_rope_index and the generation config are used, which need no weights.
        with torch.device("meta"):
            self.vlm = Cosmos3EdgeForConditionalGeneration._from_config(config.vlm_config)
        self.anchor_token = config.traj_ids["future_start"]
        self.max_new_tokens = max(256, config.tokens_per_future_traj)

    def blocked_token_ids(self) -> list[int]:
        """Tokens the reasoning may not contain, as in the model's own sampler."""
        ids = self.config.traj_ids
        blocked = list(
            range(ids["history_id0"], ids["history_id0"] + self.config.history_vocab_size)
        )
        blocked += range(ids["future_id0"], ids["future_id0"] + self.config.future_vocab_size)
        eos = self.vlm.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        blocked += [t for t in eos if t != self.anchor_token]
        return blocked

    def prepare(self, data: dict[str, Any]) -> Prompt:
        prepared = prepare_model_inputs(data, self.config, self.tokenizer)
        td = prepared["tokenized_data"]
        history = {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")}
        tokens = fuse_traj_tokens(
            self.history_tokenizer,
            self.future_tokenizer,
            td["input_ids"],
            history,
            self.config.traj_ids,
        )
        positions, rope_delta = self.vlm.model.get_rope_index(
            tokens,
            mm_token_type_ids=td["mm_token_type_ids"],
            image_grid_thw=td["image_grid_thw"],
        )
        grid = td["image_grid_thw"]
        patches = torch.split(td["pixel_values"], grid.prod(-1).tolist())
        return Prompt(
            tokens=tokens,
            positions=positions,
            rope_delta=int(rope_delta.flatten()[0]),
            patches=list(patches),
            grids=[(int(h), int(w)) for _, h, w in grid.tolist()],
            history_xyz=history["ego_history_xyz"][:, -1],
            history_rot=history["ego_history_rot"][:, -1],
        )

    def trajectory(self, action: torch.Tensor, prompt: Prompt) -> tuple[torch.Tensor, torch.Tensor]:
        """[1, 64, 2] action -> ([1, 64, 3] positions, [1, 64, 3, 3] rotations)."""
        return self.action_space.action_to_traj(
            action.float(), prompt.history_xyz, prompt.history_rot
        )

    def reasoning_text(self, generated: list[int]) -> dict[str, str]:
        decoded = extract_text_tokens(self.tokenizer, torch.tensor([generated]))
        return {k: str(v[0]) for k, v in decoded.items()}


def synthetic_sample(seed: int = 0, height: int = 540, width: int = 960) -> dict[str, Any]:
    """Smooth random camera frames with the camera and timestamp layout the model expects.

    Cameras 1 and 6 carry 5 frames over 2 seconds, the other four 3 frames over
    1 second. The ego car drives straight at 8 m/s.
    """
    g = torch.Generator().manual_seed(seed)
    cameras, stamps = [], []
    t0 = 1_700_000_000_000_000
    for cam in (0, 1, 2, 3, 5, 6):
        if cam in (1, 6):
            offsets = [-2_000_000, -1_000_000, -500_000, -100_000, 0]
        else:
            offsets = [-1_000_000, -100_000, 0]
        cameras += [cam] * len(offsets)
        stamps += [t0 + o for o in offsets]
    low = torch.rand(len(cameras), 3, 9, 16, generator=g)
    frames = torch.nn.functional.interpolate(
        low, size=(height, width), mode="bicubic", align_corners=False
    )
    frames = (frames.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(1).contiguous()
    t = torch.arange(-19, 1, dtype=torch.float32) * 0.1
    xyz = torch.stack([8.0 * t, torch.zeros_like(t), torch.zeros_like(t)], -1)
    return {
        "image_frames": frames,
        "camera_indices": torch.tensor(cameras, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(stamps, dtype=torch.int64),
        "ego_history_xyz": xyz[None],
        "ego_history_rot": torch.eye(3).expand(20, 3, 3).contiguous()[None],
        "nav_text": ["Continue straight"],
    }
