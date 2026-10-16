# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulate streaming prefill: older images reuse the K/V of the tick where they were new.

At 10 Hz, consecutive prompts share most images: each camera's newest frame is the
only new one, and every other frame reappears at a later offset (-0.1, -0.5, -1 and
-2 seconds). A streaming runtime therefore recomputes only the text and the newest
images, and splices each older image's K/V, cached per image when that image was
new, into the KV cache of the current prompt. That cached K/V was computed on the
context of an earlier tick, which was itself partly cached, so it is stale, and the
staleness compounds from tick to tick. Positions stay exact: a reused key is rotated
to the image's current slot.

``StreamingPrefill`` reproduces that function in PyTorch. It runs the whole prompt,
but forward hooks on every language-model layer replace the older images' rows of
``k_norm``'s output (K before RoPE) and ``v_proj``'s output (V) with the cached
values. The layer then applies RoPE at the current positions. Tokens interact only
through attention K/V, so once those rows are replaced, the older images' own
computation reaches no other token: the result is the same function as computing
only the recomputed rows against the spliced cache. The simulation reproduces the
semantics, not the speed: the full prompt and every image still run through the
model.

Usage, with one instance per stream and consecutive 0.1 s ticks::

    stream = StreamingPrefill(model)
    for sample in samples[:-1]:
        prepared = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), "cuda")
        stream.advance(sample, prepared)  # prefill only
    with stream.tick(samples[-1], prepared_t0) as plan:
        model.sample_trajectories_from_data(prepared_t0, ...)

Frames are identified by ``image_frame_ids``: ``(stream, camera, capture time)``.
The first ticks of a stream find nothing in the cache and recompute every image; the
cache holds every image of the last two seconds after a two-second warm-up. Run the
model eagerly: the hooks do not reach into ``alpamayo2_edge.accel``'s compiled
graphs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from alpamayo2_edge.accel.vit_cache import image_frame_ids

HISTORY_US = 2_000_000
TICK_US = 100_000


@dataclass(frozen=True)
class FrameKV:
    """One image's K (before RoPE) and V rows, one tensor per language-model layer."""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))


class ImageKVCache:
    """Per-image K/V keyed by ``(stream, camera, capture time in µs)``."""

    def __init__(self) -> None:
        self._frames: dict[Hashable, FrameKV] = {}

    def __contains__(self, frame_id: Hashable) -> bool:
        return frame_id in self._frames

    def __getitem__(self, frame_id: Hashable) -> FrameKV:
        return self._frames[frame_id]

    def __len__(self) -> int:
        return len(self._frames)

    def frame_ids(self) -> list[Hashable]:
        return list(self._frames)

    def put(self, frame_id: Hashable, frame: FrameKV) -> None:
        self._frames[frame_id] = frame

    def evict_before(self, time_us: int) -> list[Hashable]:
        """Drop the frames captured before ``time_us``; return their IDs."""
        evicted = [frame_id for frame_id in self._frames if frame_id[-1] < time_us]
        for frame_id in evicted:
            del self._frames[frame_id]
        return evicted

    def clear(self) -> None:
        self._frames.clear()

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self._frames.values())


@dataclass(frozen=True)
class TickPlan:
    """Which images of one prompt are spliced from the cache and which are recomputed."""

    frame_ids: tuple[Hashable, ...]
    spans: tuple[tuple[int, int], ...]
    reused: tuple[int, ...]
    recomputed: tuple[int, ...]
    newest: tuple[int, ...]
    prompt_tokens: int

    @property
    def reused_tokens(self) -> int:
        return sum(self.spans[i][1] - self.spans[i][0] for i in self.reused)

    @property
    def recomputed_tokens(self) -> int:
        return self.prompt_tokens - self.reused_tokens

    @property
    def missing(self) -> tuple[int, ...]:
        """Older images recomputed because the cache did not hold them (cold start)."""
        return tuple(i for i in self.recomputed if i not in self.newest)

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "recomputed_tokens": self.recomputed_tokens,
            "reused_tokens": self.reused_tokens,
            "images": len(self.frame_ids),
            "reused_images": len(self.reused),
            "recomputed_images": len(self.recomputed),
            "missing_images": [list(self.frame_ids[i][1:]) for i in self.missing],
        }


def image_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of image tokens in a one-sample prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("image_spans expects one prompt of shape [1, L]")
    mask = torch.cat([input_ids[0].eq(image_token_id), input_ids.new_zeros(1, dtype=torch.bool)])
    edges = torch.diff(mask.int(), prepend=mask.new_zeros(1, dtype=torch.int)).nonzero()[:, 0]
    bounds = edges.tolist()
    return list(zip(bounds[0::2], bounds[1::2], strict=True))


class _SpliceHooks:
    """Capture or replace image rows of K (before RoPE) and V in a prompt forward.

    Rows are touched only in forwards that start the sequence (an empty cache) and
    cover every image, the prompt prefill. Decoding steps pass through unchanged.
    """

    def __init__(self, model, plan: TickPlan, cache: ImageKVCache, capture: bool) -> None:
        self.text = model.vlm.model.language_model
        self.plan = plan
        self.capture = capture
        self.end = max((end for _, end in plan.spans), default=0)
        self.reused_rows = None
        self.sources: list[list[torch.Tensor]] = [[], []]
        if plan.reused:
            self.reused_rows = torch.cat([torch.arange(*plan.spans[i]) for i in plan.reused])
            for i in plan.reused:
                frame = cache[plan.frame_ids[i]]
                self.sources[0].append(frame.keys)
                self.sources[1].append(frame.values)
        layers = len(self.text.layers)
        self.captured: list[list[list[torch.Tensor | None]]] = [
            [[None] * layers for _ in plan.recomputed] for _ in range(2)
        ]
        self.armed = False
        self.prompt_forwards = 0
        self.handles: list[Any] = []

    def _arm(self, _module, _args, kwargs) -> None:
        cache = kwargs.get("past_key_values")
        embeds = kwargs.get("inputs_embeds")
        length = embeds.shape[1] if embeds is not None else kwargs["input_ids"].shape[1]
        self.armed = (cache is None or cache.get_seq_length() == 0) and length >= self.end
        self.prompt_forwards += self.armed

    def _disarm(self, *_args) -> None:
        self.armed = False

    def _hook(self, layer: int, kind: int):
        def apply(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
            if not self.armed:
                return output
            if self.capture:
                for slot, i in enumerate(self.plan.recomputed):
                    start, end = self.plan.spans[i]
                    self.captured[kind][slot][layer] = output[:, start:end].detach().clone()
            if self.reused_rows is None:
                return output
            cached = torch.cat([frame[layer] for frame in self.sources[kind]], dim=1)
            cached = cached.to(device=output.device, dtype=output.dtype)
            if cached.shape[2:] != output.shape[2:]:
                raise ValueError(
                    f"cached layer {layer} rows have shape {tuple(cached.shape)}, "
                    f"the prompt's have {tuple(output.shape)}"
                )
            rows = self.reused_rows.to(output.device)
            return output.index_copy(1, rows, cached.expand(output.shape[0], *cached.shape[1:]))

        return apply

    def __enter__(self) -> _SpliceHooks:
        self.handles.append(self.text.register_forward_pre_hook(self._arm, with_kwargs=True))
        self.handles.append(self.text.register_forward_hook(self._disarm))
        for index, layer in enumerate(self.text.layers):
            attention = layer.self_attn
            self.handles.append(attention.k_norm.register_forward_hook(self._hook(index, 0)))
            self.handles.append(attention.v_proj.register_forward_hook(self._hook(index, 1)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.armed = False

    def frames(self) -> dict[Hashable, FrameKV]:
        """The captured K/V of every recomputed image, if a prompt forward ran."""
        result = {}
        for slot, i in enumerate(self.plan.recomputed):
            keys, values = self.captured[0][slot], self.captured[1][slot]
            if any(t is None for t in (*keys, *values)):
                raise RuntimeError("a prompt forward did not reach every layer")
            result[self.plan.frame_ids[i]] = FrameKV(tuple(keys), tuple(values))
        return result


@dataclass
class StreamingPrefill:
    """Replay a stream tick by tick, as a runtime with a per-image K/V cache computes it.

    ``history_us`` is the camera history of a prompt: after each tick, frames older
    than the tick time minus ``history_us`` can no longer appear and are evicted.
    ``stream`` names the stream for ``image_frame_ids`` when samples carry no
    ``clip_id``.
    """

    model: Any
    history_us: int = HISTORY_US
    stream: Hashable | None = None
    cache: ImageKVCache = field(default_factory=ImageKVCache)

    def reset(self) -> None:
        self.cache.clear()

    def plan(self, sample: Mapping[str, Any], prepared: Mapping[str, Any]) -> TickPlan:
        """Match the prompt's images with the cache, without running the model."""
        tokenized, _ = self.model.prepare_tokenized_data(prepared)
        input_ids = tokenized["input_ids"]
        spans = image_spans(input_ids, self.model.config.vlm_config.image_token_id)
        frame_ids = image_frame_ids(sample, self.stream)
        grids = tokenized["image_grid_thw"]
        merge = self.model.vlm.model.visual.spatial_merge_size
        lengths = (grids.prod(-1) // merge**2).tolist()
        if len(spans) != len(frame_ids) or lengths != [end - start for start, end in spans]:
            raise ValueError(
                f"{len(spans)} image spans of {[e - s for s, e in spans]} tokens do not match "
                f"{len(frame_ids)} frames of {lengths} tokens"
            )
        latest: dict[Any, int] = {}
        for frame_id in frame_ids:
            latest[frame_id[1]] = max(latest.get(frame_id[1], frame_id[2]), frame_id[2])
        reused = tuple(i for i, frame_id in enumerate(frame_ids) if frame_id in self.cache)
        return TickPlan(
            frame_ids=tuple(frame_ids),
            spans=tuple(spans),
            reused=reused,
            recomputed=tuple(i for i in range(len(frame_ids)) if i not in reused),
            newest=tuple(
                i for i, frame_id in enumerate(frame_ids) if frame_id[2] == latest[frame_id[1]]
            ),
            prompt_tokens=int(input_ids.shape[1]),
        )

    @contextlib.contextmanager
    def tick(
        self, sample: Mapping[str, Any], prepared: Mapping[str, Any], *, reuse: bool = True
    ) -> Iterator[TickPlan]:
        """Splice cached images into the prompt forwards inside the block, then cache the rest.

        Any inference that prefills ``prepared`` may run inside the block, for example
        ``sample_trajectories_from_data``. On a clean exit the recomputed images' K/V
        enter the cache and frames that fell out of the history are evicted. With
        ``reuse=False`` every image is recomputed, as in a full prefill.
        """
        plan = self.plan(sample, prepared)
        if not reuse:
            plan = TickPlan(
                plan.frame_ids,
                plan.spans,
                (),
                tuple(range(len(plan.frame_ids))),
                plan.newest,
                plan.prompt_tokens,
            )
        hooks = _SpliceHooks(self.model, plan, self.cache, capture=True)
        with hooks:
            yield plan
        if hooks.prompt_forwards == 0:
            raise RuntimeError("no forward of the whole prompt ran inside StreamingPrefill.tick")
        for frame_id, frame in hooks.frames().items():
            self.cache.put(frame_id, frame)
        tick_us = max(frame_id[2] for frame_id in plan.frame_ids)
        self.cache.evict_before(tick_us - self.history_us)

    def advance(self, sample: Mapping[str, Any], prepared: Mapping[str, Any]) -> TickPlan:
        """Run one tick's prompt prefill and update the cache, without decoding."""
        tokenized, _ = self.model.prepare_tokenized_data(prepared)
        vision_keys = ("pixel_values", "image_grid_thw")
        inputs = {key: tokenized[key] for key in vision_keys}
        dtype = self.model.dtype
        with (
            self.tick(sample, prepared) as plan,
            torch.inference_mode(),
            torch.autocast(self.model.device.type, dtype=dtype, enabled=dtype != torch.float32),
        ):
            self.model.vlm.model(
                input_ids=tokenized["input_ids"][:, :-1],
                attention_mask=tokenized["attention_mask"][:, :-1],
                mm_token_type_ids=tokenized["mm_token_type_ids"][:, :-1],
                use_cache=False,
                **inputs,
            )
        return plan


def kv_difference(
    reference: ImageKVCache, other: ImageKVCache, frame_ids: Sequence[Hashable]
) -> list[dict[str, Any]]:
    """Per layer and kind: cosine, relative L1 and relative L2 of ``other`` over ``frame_ids``.

    Cosine separates direction from magnitude: a high cosine with a large relative L2
    is a scale difference, a low cosine a structural one.
    """
    rows = []
    if not frame_ids:
        return rows
    layers = len(reference[frame_ids[0]].keys)
    for kind in ("keys", "values"):
        for layer in range(layers):
            ref = torch.cat(
                [getattr(reference[f], kind)[layer].float().flatten() for f in frame_ids]
            )
            val = torch.cat([getattr(other[f], kind)[layer].float().flatten() for f in frame_ids])
            diff = val - ref
            rows.append(
                {
                    "kind": kind[0],
                    "layer": layer,
                    "cos": float(torch.nn.functional.cosine_similarity(ref, val, dim=0)),
                    "rel_l1": float(diff.abs().sum() / ref.abs().sum().clamp_min(1e-12)),
                    "rel_l2": float(diff.norm() / ref.norm().clamp_min(1e-12)),
                }
            )
    return rows


def stream_ticks_us(t0_us: int, warmup_us: int, tick_us: int = TICK_US) -> list[int]:
    """Tick times from a cold start ``warmup_us`` before ``t0_us`` up to ``t0_us``."""
    if warmup_us < 0 or tick_us <= 0 or warmup_us % tick_us:
        raise ValueError(f"warm-up {warmup_us} us must be a whole number of {tick_us} us ticks")
    return [t0_us - warmup_us + index * tick_us for index in range(warmup_us // tick_us + 1)]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _kv_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        kind: {
            metric: {
                "median": _median([row[metric] for row in rows if row["kind"] == kind]),
                "worst": (min if metric == "cos" else max)(
                    row[metric] for row in rows if row["kind"] == kind
                ),
            }
            for metric in ("cos", "rel_l1", "rel_l2")
        }
        for kind in ("k", "v")
        if any(row["kind"] == kind for row in rows)
    }


def main(argv=None) -> None:
    """Compare streamed and full-prefill inference at one time of a PhysicalAI-AV clip."""
    from alpamayo2_edge.inference import _clip_spec, load_clip_manifest, predict

    parser = argparse.ArgumentParser(description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--clip-id")
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--t0-us", type=int)
    parser.add_argument("--nav-text")
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=3.0,
        help="Start the stream cold this many seconds before t0 (at least 2 s fills the cache)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="1 decodes greedily, so the two runs differ only by the prefix (default)",
    )
    parser.add_argument("--diffusion-steps", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.manifest:
            clip = load_clip_manifest(args.manifest, args.sample_index or 0, args.nav_text)
        else:
            clip = _clip_spec(
                args.clip_id, 5_100_000 if args.t0_us is None else args.t0_us, args.nav_text
            )
        warmup_us = round(args.warmup_s * 1_000_000)
        ticks = stream_ticks_us(clip["t0_us"], warmup_us)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if ticks[0] < HISTORY_US:
        parser.error(f"the cold start at {ticks[0]} us has no two-second camera history")

    import physical_ai_av

    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

    model, _ = load_model(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def load(tick_us: int) -> tuple[dict, dict]:
        sample = load_physical_aiavdataset(
            clip["clip_id"], tick_us, nav_text=clip["nav_text"], avdi=avdi
        )
        prepared = prepare_model_inputs(sample, model.config, model.tokenizer)
        return sample, to_device(prepared, device=args.device)

    settings = dict(
        seed=args.seed,
        top_p=0.98,
        top_k=args.top_k,
        temperature=0.6,
        num_traj_sets=1,
        num_traj_samples=1,
        max_generation_length=256,
        diffusion_kwargs={"inference_step": args.diffusion_steps},
    )
    stream = StreamingPrefill(model)
    tick_reports = []
    for tick_us in ticks[:-1]:
        started = time.perf_counter()
        plan = stream.advance(*load(tick_us))
        tick_reports.append(
            {"tick_us": tick_us, "wall_s": time.perf_counter() - started, **plan.summary()}
        )
    sample, prepared = load(clip["t0_us"])
    with stream.tick(sample, prepared) as plan:
        streamed = predict(model, prepared, settings, args.device)
    reference = StreamingPrefill(model)
    with reference.tick(sample, prepared, reuse=False):
        full = predict(model, prepared, settings, args.device)

    older = [plan.frame_ids[i] for i in plan.reused]
    newest = [plan.frame_ids[i] for i in plan.newest]
    distance = (streamed[0] - full[0])[..., :2].float().norm(dim=-1).reshape(-1)
    report = {
        "input_source": clip,
        "warmup_s": args.warmup_s,
        "settings": settings,
        "ticks": tick_reports,
        "t0": plan.summary(),
        "cot": {
            "streamed": streamed[2]["cot"].reshape(-1).tolist(),
            "full_prefill": full[2]["cot"].reshape(-1).tolist(),
        },
        "trajectory_streamed_vs_full_m": {
            "ade": float(distance.mean()),
            "fde": float(distance[-1]),
        },
        "kv_vs_full_prefill": {
            "reused_images": _kv_summary(kv_difference(reference.cache, stream.cache, older)),
            "newest_images": _kv_summary(kv_difference(reference.cache, stream.cache, newest)),
        },
        "cache_bytes": stream.cache.nbytes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "streaming_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("t0", "trajectory_streamed_vs_full_m")}))


if __name__ == "__main__":
    main()
