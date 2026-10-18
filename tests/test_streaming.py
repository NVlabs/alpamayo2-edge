"""Streaming-prefill simulation: per-image K/V reuse on a tiny real Qwen3-VL."""

import pytest
import torch

from alpamayo2_edge.streaming import (
    FrameKV,
    ImageKVCache,
    StreamingPrefill,
    image_spans,
    kv_difference,
    stream_ticks_us,
)

ANCHOR = 9
PATCH_DIM = 3 * 2 * 2 * 2
TICK = 100_000
# Camera 1 keeps three frames, camera 6 two, as the front and side cameras do.
SCHEDULE = {1: ((1, 4, 4), (-200_000, -100_000, 0)), 6: ((1, 4, 6), (-100_000, 0))}


def _tiny_model(monkeypatch):
    from transformers import Qwen3VLConfig

    import alpamayo2_edge.models.alpamayo2_edge as module
    from alpamayo2_edge.config import Alpamayo2EdgeConfig

    text_config = {
        "model_type": "qwen3_vl_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
            "mrope_interleaved": True,
        },
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    config = Alpamayo2EdgeConfig(
        dtype="float32",
        vlm_name_or_path="test-tokenizer",
        vlm_config=Qwen3VLConfig(
            text_config=text_config,
            image_token_id=10,
            video_token_id=13,
            vision_start_token_id=11,
            vision_end_token_id=12,
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_heads": 2,
                "out_hidden_size": 64,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0],
            },
        ),
        expert_config={
            "llm_config": {**text_config, "hidden_size": 32},
            "action_space_cfg": {
                "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureActionSpace",
                "n_waypoints": 4,
            },
            "diffusion_cfg": {"_target_": "alpamayo2_edge.diffusion.flow_matching.FlowMatching"},
            "action_in_proj_cfg": {
                "_target_": "alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2",
                "hidden_size": 32,
                "num_enc_layers": 1,
            },
            "action_out_proj_cfg": {"_target_": "torch.nn.Linear"},
        },
        traj_ids={"history_id0": 100, "future_id0": 110, "future_start": ANCHOR},
        history_vocab_size=10,
        future_vocab_size=10,
    )
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "pad_token_id": 0,
            "batch_decode": lambda self, ids, **kwargs: [
                " ".join(map(str, row.tolist())) for row in ids
            ],
        },
    )()
    monkeypatch.setattr(module, "build_alpamayo2_edge_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(module, "fuse_traj_tokens", lambda a, b, ids, data, config: ids)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = module.Alpamayo2Edge(config).eval()
    # Greedy decoding then emits token t + 1 after token t, a CoT of known length.
    text = model.vlm.model.language_model
    hidden = text.config.hidden_size
    with torch.no_grad():
        text.embed_tokens.weight.mul_(0.1)
        text.embed_tokens.weight[:hidden] += 100 * torch.eye(hidden)
        model.vlm.lm_head.weight.zero_()
        model.vlm.lm_head.weight[1 : hidden + 1] = torch.eye(hidden)
    return model


def _frame_pixels(camera: int, time_us: int, grid) -> torch.Tensor:
    generator = torch.Generator().manual_seed(camera * 10_000_019 + time_us // 1000)
    return torch.randn(grid[0] * grid[1] * grid[2], PATCH_DIM, generator=generator)


def _tick(t_us: int, noise_before: int | None = None, early_us: int = 0):
    """A raw sample (frame IDs) and model inputs whose frame pixels depend only on the frame.

    With ``noise_before``, frames captured before that time get unrelated pixels.
    ``early_us`` makes camera 1's captures precede their requested times, as decoded
    frames do; camera 6's newest frame stays at the tick time.
    """
    cameras, times, grids, pixels = [], [], [], []
    ids = [1]
    for camera, (grid, offsets) in SCHEDULE.items():
        for offset in offsets:
            time_us = t_us + offset - (early_us if camera == 1 else 0)
            cameras.append(camera)
            times.append(time_us)
            grids.append(grid)
            if noise_before is not None and time_us < noise_before:
                pixels.append(torch.randn(grid[0] * grid[1] * grid[2], PATCH_DIM))
            else:
                pixels.append(_frame_pixels(camera, time_us, grid))
            ids += [11] + [10] * (grid[1] * grid[2] // 4) + [12]
    ids += [3, 4, 5]
    input_ids = torch.tensor([ids])
    sample = {
        "clip_id": "stream",
        "camera_indices": torch.tensor(cameras),
        "absolute_timestamps": torch.tensor(times),
    }
    prepared = {
        "tokenized_data": {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": input_ids.eq(10).long(),
            "pixel_values": torch.cat(pixels),
            "image_grid_thw": torch.tensor(grids),
        },
        "ego_history_xyz": torch.zeros(1, 1, 20, 3),
        "ego_history_rot": torch.eye(3).expand(1, 1, 20, 3, 3).clone(),
    }
    return sample, prepared


def _sample(model, prepared):
    with torch.inference_mode():
        xyz, _, _, extra = model.sample_trajectories_from_data(
            prepared,
            top_k=1,
            max_generation_length=6,
            diffusion_kwargs={"inference_step": 2, "generator": torch.Generator().manual_seed(3)},
            return_extra=True,
        )
    return xyz, extra["generated_ids"]


def _prompt_hidden(model, prepared):
    tokenized, _ = model.prepare_tokenized_data(prepared)
    with torch.inference_mode():
        return model.vlm.model(**tokenized, use_cache=False).last_hidden_state


def _assert_same(first, second):
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_image_spans_and_ticks():
    ids = torch.tensor([[1, 10, 10, 3, 10, 2, 10, 10, 10]])
    assert image_spans(ids, 10) == [(1, 3), (4, 5), (6, 9)]
    assert stream_ticks_us(3_000_000, 300_000) == [2_700_000, 2_800_000, 2_900_000, 3_000_000]
    with pytest.raises(ValueError, match="whole number"):
        stream_ticks_us(3_000_000, 250_000)


def test_cold_tick_is_the_full_prefill_and_fills_the_cache(monkeypatch):
    model = _tiny_model(monkeypatch)
    sample, prepared = _tick(1_000_000)
    stream = StreamingPrefill(model)
    with stream.tick(sample, prepared) as plan:
        streamed = _sample(model, prepared)
    _assert_same(streamed, _sample(model, prepared))
    assert plan.reused == () and plan.missing == (0, 1, 3)
    assert plan.recomputed_tokens == plan.prompt_tokens
    assert sorted(stream.cache.frame_ids()) == sorted(plan.frame_ids)
    frame = stream.cache[plan.frame_ids[0]]
    assert len(frame.keys) == 3 and frame.keys[0].shape == (1, 4, 2, 16)
    assert frame.values[0].shape == (1, 4, 32)


def test_splicing_a_tick_into_itself_is_exact(monkeypatch):
    model = _tiny_model(monkeypatch)
    sample, prepared = _tick(1_000_000)
    stream = StreamingPrefill(model)
    stream.advance(sample, prepared)
    with stream.tick(sample, prepared) as plan:
        spliced = _sample(model, prepared)
    assert len(plan.reused) == 5 and plan.recomputed_tokens == plan.prompt_tokens - 24
    _assert_same(spliced, _sample(model, prepared))


def test_reused_images_reach_no_other_token(monkeypatch):
    """Their own rows are overwritten in every layer, so their pixels cannot matter."""
    model = _tiny_model(monkeypatch)
    sample, prepared = _tick(1_000_000)
    _, noisy = _tick(1_000_000, noise_before=1_000_000)
    stream = StreamingPrefill(model)
    stream.advance(sample, prepared)
    with stream.tick(sample, noisy) as plan:
        with_noise = _sample(model, noisy)
        hidden = _prompt_hidden(model, noisy)
    _assert_same(with_noise, _sample(model, prepared))
    other_rows = torch.ones(plan.prompt_tokens, dtype=torch.bool)
    for i in plan.reused:
        other_rows[slice(*plan.spans[i])] = False
    clean = _prompt_hidden(model, prepared)
    assert torch.equal(hidden[:, other_rows], clean[:, other_rows])
    # The harness is sensitive: without the splice the same noise changes those rows.
    assert not torch.equal(_prompt_hidden(model, noisy)[:, other_rows], clean[:, other_rows])


def test_stream_reuses_older_frames_with_stale_context(monkeypatch):
    model = _tiny_model(monkeypatch)
    stream = StreamingPrefill(model)
    stream.advance(*_tick(1_000_000))
    sample, prepared = _tick(1_100_000)
    plan = stream.advance(sample, prepared)
    reused = sorted(plan.frame_ids[i][1:] for i in plan.reused)
    assert reused == [(1, 900_000), (1, 1_000_000), (6, 1_000_000)]
    assert sorted(plan.frame_ids[i][1:] for i in plan.recomputed) == [
        (1, 1_100_000),
        (6, 1_100_000),
    ]
    assert plan.missing == ()
    assert plan.recomputed_tokens == plan.prompt_tokens - (4 + 4 + 6)
    # After the tick at 1.1 s, frames before 0.9 s can no longer appear.
    assert ("stream", 1, 800_000) not in stream.cache
    assert len(stream.cache) == 6

    reference = StreamingPrefill(model)
    reference.advance(sample, prepared)
    older = [plan.frame_ids[i] for i in plan.reused]
    rows = kv_difference(reference.cache, stream.cache, older)
    by_layer = {(row["kind"], row["layer"]): row for row in rows}
    # The first layer's K/V depend on the image alone; deeper layers saw another context.
    assert by_layer["k", 0]["rel_l2"] < 1e-5 and by_layer["v", 0]["rel_l2"] < 1e-5
    assert by_layer["k", 2]["rel_l2"] > 1e-4 and by_layer["v", 2]["rel_l2"] > 1e-4


def test_cached_keys_precede_rope(monkeypatch):
    """A frame's first-layer K is the same in any slot, so positions come from the current tick."""
    model = _tiny_model(monkeypatch)
    first = StreamingPrefill(model)
    first.advance(*_tick(1_000_000))
    second = StreamingPrefill(model)
    sample, prepared = _tick(1_100_000)
    plan = second.advance(sample, prepared)
    frame_id = ("stream", 1, 1_000_000)
    first_slot = first.plan(*_tick(1_000_000)).frame_ids.index(frame_id)
    assert plan.frame_ids.index(frame_id) != first_slot
    torch.testing.assert_close(
        first.cache[frame_id].keys[0], second.cache[frame_id].keys[0], rtol=0, atol=1e-6
    )


def test_captures_that_precede_their_request_stay_cached(monkeypatch):
    """Decoded frames can be slightly older than requested; no frame in use is evicted."""
    model = _tiny_model(monkeypatch)
    stream = StreamingPrefill(model)
    for t_us in (1_000_000, 1_100_000):
        stream.advance(*_tick(t_us, early_us=7_000))
    sample, prepared = _tick(1_200_000, early_us=7_000)
    with stream.tick(sample, prepared) as plan:
        _prompt_hidden(model, prepared)
    assert plan.missing == () and len(plan.reused) == 3
    assert all(frame_id in stream.cache for frame_id in plan.frame_ids)
    assert ("stream", 1, 893_000) not in stream.cache


def test_gaps_recompute_the_missing_frames(monkeypatch):
    model = _tiny_model(monkeypatch)
    stream = StreamingPrefill(model)
    stream.advance(*_tick(1_000_000))
    plan = stream.advance(*_tick(1_200_000))
    assert [plan.frame_ids[i][1:] for i in plan.missing] == [(1, 1_100_000), (6, 1_100_000)]
    assert [plan.frame_ids[i][1:] for i in plan.reused] == [(1, 1_000_000)]
    assert ("stream", 6, 1_100_000) in stream.cache


def test_a_tick_without_a_prompt_forward_is_an_error(monkeypatch):
    model = _tiny_model(monkeypatch)
    stream = StreamingPrefill(model)
    with pytest.raises(RuntimeError, match="no forward"):
        with stream.tick(*_tick(1_000_000)):
            pass
    assert len(stream.cache) == 0


def test_cli_reports_streamed_against_full_prefill(monkeypatch, tmp_path):
    import json
    import sys
    import types

    import alpamayo2_edge.checkpoint as checkpoint
    import alpamayo2_edge.helper as helper
    import alpamayo2_edge.load_physical_aiavdataset as loader
    from alpamayo2_edge import streaming

    model = _tiny_model(monkeypatch)
    loads = []

    def load_sample(clip_id, t0_us, nav_text, avdi):
        loads.append(t0_us)
        sample, _ = _tick(t0_us)
        return {**sample, "clip_id": clip_id, "t0_us": t0_us}

    monkeypatch.setattr(checkpoint, "load_model", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(loader, "load_physical_aiavdataset", load_sample)
    monkeypatch.setattr(helper, "prepare_model_inputs", lambda data, *args: _tick(data["t0_us"])[1])
    fake_sdk = types.ModuleType("physical_ai_av")
    fake_sdk.PhysicalAIAVDatasetInterface = lambda: None
    monkeypatch.setitem(sys.modules, "physical_ai_av", fake_sdk)

    streaming.main(
        [
            "--model-id=unused",
            "--clip-id=ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
            "--t0-us=3000000",
            "--nav-text=Follow the road",
            "--warmup-s=0.3",
            "--device=cpu",
            "--dtype=float32",
            "--diffusion-steps=2",
            f"--output={tmp_path}",
        ]
    )
    report = json.loads((tmp_path / "streaming_report.json").read_text())
    assert loads == [2_700_000, 2_800_000, 2_900_000, 3_000_000]
    assert [tick["reused_images"] for tick in report["ticks"]] == [0, 3, 3]
    assert report["t0"]["reused_images"] == 3 and report["t0"]["missing_images"] == []
    assert set(report["kv_vs_full_prefill"]) == {"reused_images", "newest_images"}
    assert report["kv_vs_full_prefill"]["reused_images"]["k"]["cos"]["worst"] < 1
    assert report["trajectory_streamed_vs_full_m"]["ade"] >= 0


def test_kv_difference_metrics():
    reference, other = ImageKVCache(), ImageKVCache()
    keys = torch.tensor([[[[3.0, 4.0]]]])
    reference.put("f", FrameKV((keys,), (keys,)))
    other.put("f", FrameKV((2 * keys,), (-keys,)))
    rows = {row["kind"]: row for row in kv_difference(reference, other, ["f"])}
    assert rows["k"]["cos"] == pytest.approx(1.0)
    assert rows["k"]["rel_l2"] == pytest.approx(1.0)
    assert rows["v"]["cos"] == pytest.approx(-1.0)
    assert rows["v"]["rel_l1"] == pytest.approx(2.0)
