# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse vision-tower features of frames seen in earlier calls (streaming inference).

In streaming use, consecutive calls share most frames: a frame that is current at
one time step reappears at its past offsets (-0.1, -0.5, -1 and -2 seconds) in
later prompts. Qwen3-VL encodes images independently, so a frame's merged features
and DeepStack features do not depend on the other images in the call. The cache
stores them per frame and encodes only frames it has not seen, normally the
newest frame of each camera.

Frames are identified by the caller (``image_frame_ids`` gives ``(camera, capture
time in microseconds)`` from a raw sample) together with the frame's patch grid
and a fingerprint of its pixels. A reused feature therefore requires the same
frame identity, resolution and sampled pixel content. ``full_digest=True`` hashes
every pixel instead of a strided sample, at about 2 ms of CPU time per image.

Usage, with one cache per stream (call ``clear()`` when the stream changes)::

    cache = VisionFeatureCache(model)
    for sample in samples:  # consecutive time steps
        prepared = prepare_model_inputs(sample, model.config, model.tokenizer)
        prepared = cache.prepare(prepared, image_frame_ids(sample))
        inputs = to_device(prepared, "cuda")
        model.sample_trajectories_from_data(inputs, ...)
"""

import hashlib
import zlib
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from alpamayo2_edge.accel.graphs import override_forward
from alpamayo2_edge.sample import canonical_frame_order

Features = tuple[torch.Tensor, tuple[torch.Tensor, ...]]  # merged rows, DeepStack rows


def image_frame_ids(sample: Mapping[str, Any]) -> list[tuple[int, int]]:
    """``(camera, capture time in µs)`` of each image of a raw sample, in ``pixel_values`` order.

    ``prepare_model_inputs`` packs images sorted by camera, then capture time
    (``canonical_frame_order``).
    """
    cameras, times = sample["camera_indices"], sample["absolute_timestamps"].reshape(-1)
    order = canonical_frame_order(cameras, times)
    return list(zip(cameras[order].tolist(), times[order].tolist(), strict=True))


def _fingerprint(rows: torch.Tensor, full_digest: bool) -> str:
    rows = rows.detach()
    if not full_digest:
        # A strided sample of patch rows: any other frame differs in these rows.
        index = torch.linspace(0, rows.shape[0] - 1, min(rows.shape[0], 16)).round().long()
        rows = rows.index_select(0, index.to(rows.device))
    data = rows.cpu().contiguous().view(torch.uint8).numpy()
    if full_digest:
        return hashlib.sha1(data).hexdigest()
    return f"{zlib.crc32(data):08x}"


@dataclass
class _Plan:
    """What the next vision call needs: the encoded images and the reused ones."""

    grid: list[tuple[int, int, int]]
    tokens: list[int]
    keys: list[Hashable]
    misses: list[int]
    reused: dict[int, Features] = field(default_factory=dict)
    n_patches: int = 0


class VisionFeatureCache:
    """``visual.forward`` wrapper that reuses features of previously encoded frames.

    Install it after any acceleration so that it wraps the outermost vision call.
    ``prepare`` must run on the inputs of every call; its plan is consumed by the
    next vision forward. Vision calls without a plan run uncached.
    """

    def __init__(
        self, model: torch.nn.Module, *, max_images: int = 256, full_digest: bool = False
    ) -> None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        self.visual = model.vlm.model.visual
        self.merge = int(self.visual.spatial_merge_size)
        self.max_images = max_images
        self.full_digest = full_digest
        self.store: OrderedDict[Hashable, Features] = OrderedDict()
        self.stats = {
            "calls": 0,
            "uncached_calls": 0,
            "images_encoded": 0,
            "images_reused": 0,
        }
        self._plan: _Plan | None = None
        self.inner = self.visual.forward
        self.installed = override_forward(self.inner, self._forward)
        self._had_instance_forward = "forward" in vars(self.visual)
        self.visual.forward = self.installed

    def remove(self) -> None:
        """Restore the vision forward this cache wrapped."""
        if vars(self.visual).get("forward") is not self.installed:
            raise RuntimeError("Remove wrappers installed after the ViT cache first")
        if self._had_instance_forward:
            self.visual.forward = self.inner
        else:
            del self.visual.forward

    def clear(self) -> None:
        """Drop every cached feature, for example at a stream boundary."""
        self.store.clear()
        self._plan = None

    def describe(self) -> dict[str, Any]:
        return {"max_images": self.max_images, "cached_images": len(self.store), **self.stats}

    def prepare(self, inputs: Mapping[str, Any], frame_ids: Sequence[Hashable]) -> dict[str, Any]:
        """Return ``inputs`` with patches only for frames whose features are not cached.

        ``frame_ids`` names each image of ``tokenized_data["pixel_values"]`` in order.
        """
        data = inputs["tokenized_data"]
        pixels, grid_thw = data.get("pixel_values"), data.get("image_grid_thw")
        if pixels is None or grid_thw is None or data.get("pixel_values_videos") is not None:
            raise ValueError("The ViT cache supports image inputs (pixel_values) only")
        if data["input_ids"].shape[0] != 1:
            raise ValueError("The ViT cache supports one sample per call")
        grid = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
        if len(frame_ids) != len(grid):
            raise ValueError(f"{len(frame_ids)} frame IDs for {len(grid)} images")
        patches = [t * h * w for t, h, w in grid]
        if sum(patches) != pixels.shape[0]:
            raise ValueError(
                f"image_grid_thw covers {sum(patches)} patches; pixel_values has {pixels.shape[0]}"
            )
        starts = [sum(patches[:index]) for index in range(len(grid))]
        keys, misses, reused = [], [], {}
        for index, (frame_id, row, start, count) in enumerate(
            zip(frame_ids, grid, starts, patches, strict=True)
        ):
            rows = pixels[start : start + count]
            key = (frame_id, row, str(pixels.dtype), _fingerprint(rows, self.full_digest))
            keys.append(key)
            features = self.store.get(key)
            if features is None:
                misses.append(index)
            else:
                self.store.move_to_end(key)
                reused[index] = features
        if misses:
            kept = torch.cat([pixels[starts[i] : starts[i] + patches[i]] for i in misses])
        else:
            kept = pixels[:0]
        self._plan = _Plan(
            grid=grid,
            tokens=[count // self.merge**2 for count in patches],
            keys=keys,
            misses=misses,
            reused=reused,
            n_patches=int(kept.shape[0]),
        )
        prepared = dict(inputs)
        prepared["tokenized_data"] = {**data, "pixel_values": kept}
        return prepared

    def _forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any):
        plan, self._plan = self._plan, None
        self.stats["calls"] += 1
        if plan is None:
            self.stats["uncached_calls"] += 1
            return self.inner(hidden_states, grid_thw, **kwargs)
        grid = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
        if grid != plan.grid or hidden_states.shape[0] != plan.n_patches:
            raise RuntimeError(
                "ViT cache: this vision call does not match the inputs passed to prepare()"
            )
        fresh: dict[int, Features] = {}
        if plan.misses:
            miss_grid = grid_thw.new_tensor([plan.grid[index] for index in plan.misses])
            output = self.inner(hidden_states, miss_grid, **kwargs)
            sizes = [plan.tokens[index] for index in plan.misses]
            merged = torch.split(output.pooler_output, sizes)
            deepstack = [torch.split(level, sizes) for level in output.deepstack_features]
            for position, index in enumerate(plan.misses):
                features = (
                    merged[position].clone(),
                    tuple(level[position].clone() for level in deepstack),
                )
                fresh[index] = features
                self.store[plan.keys[index]] = features
                self.store.move_to_end(plan.keys[index])
        parts = [fresh.get(index) or plan.reused[index] for index in range(len(plan.grid))]
        reference = parts[0]
        for features in parts:
            if features[0].dtype != reference[0].dtype or len(features[1]) != len(reference[1]):
                raise RuntimeError("ViT cache: cached features do not match the vision tower")
        while len(self.store) > self.max_images:
            self.store.popitem(last=False)
        self.stats["images_encoded"] += len(plan.misses)
        self.stats["images_reused"] += len(plan.grid) - len(plan.misses)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None,
            pooler_output=torch.cat([features[0] for features in parts]),
            deepstack_features=[
                torch.cat([features[1][level] for features in parts])
                for level in range(len(reference[1]))
            ],
        )
