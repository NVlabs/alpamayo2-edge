# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision tower acceleration.

Qwen3-VL encodes every image independently: attention is block-diagonal over
images, and patch embedding, position tables, MLPs, norms and the patch mergers
act per patch. ``VisionAccelerator`` therefore splits a call into groups of images
that share one patch grid (side and front cameras differ) and runs each group as
one graph replay of the tower with one batched attention call per block. Rows are
interleaved back in image order. The grid-dependent position tables come from
transformers' own helpers, computed once per grid.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    apply_rotary_pos_emb_vision,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from alpamayo2_edge.accel.graphs import (
    AccelStats,
    GraphedCall,
    autocast_state,
    maybe_compile,
    override_forward,
)
from alpamayo2_edge.accel.kernels import fa3_attention

# ``get_image_features`` passes these through ``visual(...)``; they do not change
# the computation.
_IGNORED_KWARGS = frozenset({"return_dict", "use_cache"})


class VisionTower(torch.nn.Module):
    """``Qwen3VLVisionModel.forward`` after its grid tables, for images of one grid.

    HF's SDPA path splits q/k/v per image and calls SDPA once per image; images of
    one grid are equally long, so one batched call over ``[images, heads, tokens,
    head_dim]`` computes the same attention.
    """

    def __init__(self, visual: torch.nn.Module, n_frames: int, fa3: bool) -> None:
        super().__init__()
        self.visual, self.n_frames, self.fa3 = visual, n_frames, fa3

    def _attention(self, attn, x, cos, sin):
        tokens = x.shape[0]
        q, k, v = attn.qkv(x).reshape(tokens, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        def frames(tensor):
            return tensor.reshape(self.n_frames, tokens // self.n_frames, attn.num_heads, -1)

        if self.fa3:
            out = fa3_attention(frames(q), frames(k), frames(v), attn.scaling, False)
        else:
            out = F.scaled_dot_product_attention(
                frames(q).transpose(1, 2),
                frames(k).transpose(1, 2),
                frames(v).transpose(1, 2),
                scale=attn.scaling,
            ).transpose(1, 2)
        return attn.proj(out.reshape(tokens, -1))

    def forward(self, pixel_values, pos_embeds, cos, sin):
        visual = self.visual
        hidden = visual.patch_embed(pixel_values)
        hidden = hidden + pos_embeds.to(hidden.dtype)
        deepstack = []
        for index, block in enumerate(visual.blocks):
            hidden = hidden + self._attention(block.attn, block.norm1(hidden), cos, sin)
            hidden = hidden + block.mlp(block.norm2(hidden))
            if index in visual.deepstack_visual_indexes:
                merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(index)]
                deepstack.append(merger(hidden))
        return visual.merger(hidden), deepstack


def grid_tables(visual: torch.nn.Module, grid_thw: torch.Tensor, hidden: torch.Tensor):
    """Position-embedding and rotary tables, computed as ``Qwen3VLVisionModel.forward`` does."""
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    pos_embeds = (visual.pos_embed(indices) * weights[:, :, None]).sum(1)
    position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    cos, sin = visual.rotary_pos_emb(hidden, position_ids)
    return pos_embeds, cos, sin


@dataclass
class _GroupLayout:
    """One vision call split into groups of images sharing a patch grid."""

    rows: list[tuple]  # per group: the grid rows of its images
    patch_index: list[torch.Tensor]  # per group: rows of the input patches
    grids: list[torch.Tensor]  # per group: [images, 3] grid_thw
    token_order: torch.Tensor | None  # output rows back in image order (None: one group)


class VisionAccelerator:
    """``visual.forward`` override replaying one graph per (patch grid, image count)."""

    def __init__(
        self,
        visual: torch.nn.Module,
        *,
        stats: AccelStats,
        compile: bool,
        cuda_graphs: bool,
        warmup: int,
        mode: str,
        fa3: bool,
    ) -> None:
        self.visual, self.stats = visual, stats
        self.compile, self.cuda_graphs, self.warmup, self.mode = compile, cuda_graphs, warmup, mode
        self.fa3 = fa3
        self.merge = int(visual.spatial_merge_size)
        self.graphs: dict[tuple, GraphedCall] = {}
        self.layouts: dict[tuple, _GroupLayout] = {}
        self.original = visual.forward
        self.installed = override_forward(self.original, self.forward)
        visual.forward = self.installed

    def remove(self) -> None:
        if vars(self.visual).get("forward") is not self.installed:
            raise RuntimeError("Remove wrappers installed after the vision accelerator first")
        del self.visual.forward

    def _layout(self, grid: tuple, device: torch.device) -> _GroupLayout:
        layout = self.layouts.get((grid, device))
        if layout is not None:
            return layout
        patches = [t * h * w for t, h, w in grid]
        tokens = [count // self.merge**2 for count in patches]
        patch_start = [sum(patches[:index]) for index in range(len(grid))]
        token_start = [sum(tokens[:index]) for index in range(len(grid))]
        groups: dict[tuple, list[int]] = {}
        for index, row in enumerate(grid):
            groups.setdefault(row, []).append(index)
        group_rows, patch_index, grids, order = [], [], [], []
        for row, members in groups.items():
            group_rows.append((row,) * len(members))
            patch_index.append(
                torch.cat(
                    [torch.arange(patch_start[i], patch_start[i] + patches[i]) for i in members]
                ).to(device)
            )
            grids.append(torch.tensor([row] * len(members), dtype=torch.long, device=device))
            for i in members:
                order.extend(range(token_start[i], token_start[i] + tokens[i]))
        token_order = None
        if len(groups) > 1:
            order = torch.tensor(order, dtype=torch.long)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel())
            token_order = inverse.to(device)
        layout = _GroupLayout(group_rows, patch_index, grids, token_order)
        self.layouts[(grid, device)] = layout
        return layout

    def _run_group(self, hidden_states, index, grid_thw, rows):
        """Replay the graph of one uniform-grid group; ``index`` selects its patch rows."""
        key = (rows[0], len(rows), hidden_states.dtype, autocast_state(hidden_states.device.type))
        graph = self.graphs.get(key)
        if graph is None:
            hidden = hidden_states if index is None else hidden_states.index_select(0, index)
            pos_embeds, cos, sin = grid_tables(self.visual, grid_thw, hidden)
            n_frames = sum(row[0] for row in rows)
            tower = maybe_compile(
                VisionTower(self.visual, n_frames, self.fa3), self.compile, self.mode
            )
            graph = GraphedCall(
                "vision",
                (rows[0], len(rows)),
                tower,
                (hidden.clone(), pos_embeds, cos, sin),
                stats=self.stats,
                warmup=self.warmup if self.cuda_graphs else 0,
                cuda_graph=self.cuda_graphs,
                device=hidden_states.device,
            )
            self.graphs[key] = graph
        if index is None:
            graph.args[0].copy_(hidden_states)
        else:
            torch.index_select(hidden_states, 0, index, out=graph.args[0])
        return graph()

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any):
        if set(kwargs) - _IGNORED_KWARGS:
            self.stats.fallback("vision_kwargs")
            return self.original(hidden_states, grid_thw, **kwargs)
        grid = tuple(tuple(int(value) for value in row) for row in grid_thw.tolist())
        layout = self._layout(grid, hidden_states.device)
        if layout.token_order is None:
            merged, deepstack = self._run_group(hidden_states, None, grid_thw, grid)
            merged, deepstack = merged.clone(), [features.clone() for features in deepstack]
        else:
            outputs = [
                self._run_group(hidden_states, index, group_grid, rows)
                for rows, index, group_grid in zip(
                    layout.rows, layout.patch_index, layout.grids, strict=True
                )
            ]
            merged = torch.cat([output[0] for output in outputs]).index_select(
                0, layout.token_order
            )
            deepstack = [
                torch.cat([output[1][level] for output in outputs]).index_select(
                    0, layout.token_order
                )
                for level in range(len(outputs[0][1]))
            ]
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None, pooler_output=merged, deepstack_features=deepstack
        )
