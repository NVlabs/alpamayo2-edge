"""Opt-in acceleration: exactness of the rewrites and parity with the eager model."""

import copy
from types import SimpleNamespace

import pytest
import torch

from alpamayo2_edge.accel import AccelConfig, FlashKVCache, enable_acceleration, resolve_kernels
from alpamayo2_edge.accel import kernels as kernels_module
from alpamayo2_edge.accel.decode import MaskColumns
from alpamayo2_edge.accel.expert import _mask_hides_tokens

ANCHOR = 9
# Two patch grids, as the side and front cameras have: 16 and 24 patches per image.
GRIDS = [(1, 4, 4), (1, 4, 4), (1, 4, 6), (1, 4, 4), (1, 4, 6)]
PATCH_DIM = 3 * 2 * 2


def _tiny_model(monkeypatch, dtype="float32"):
    """Real Cosmos3-Edge vision/text layers and an expert, without assets."""
    from transformers import Cosmos3EdgeConfig

    import alpamayo2_edge.models.alpamayo2_edge as module
    from alpamayo2_edge.config import Alpamayo2EdgeConfig

    text_config = {
        "model_type": "cosmos3_edge_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
        },
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    config = Alpamayo2EdgeConfig(
        dtype=dtype,
        vlm_name_or_path="test-tokenizer",
        vlm_config=Cosmos3EdgeConfig(
            text_config=text_config,
            image_token_id=10,
            video_token_id=13,
            vision_start_token_id=11,
            vision_end_token_id=12,
            projector_hidden_size=48,
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "patch_size": 2,
                "num_patches": 16,
                "spatial_merge_size": 2,
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
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda ids, **kwargs: [" ".join(map(str, row.tolist())) for row in ids],
    )
    monkeypatch.setattr(module, "build_alpamayo2_edge_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(module, "fuse_traj_tokens", lambda a, b, ids, data, config: ids)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = module.Alpamayo2Edge(config).eval()
    # Greedy decoding then emits token t + 1 after token t, a CoT of known length.
    text = model.vlm.model.language_model
    hidden = text.config.hidden_size
    with torch.no_grad():
        text.embed_tokens.weight.zero_()
        text.embed_tokens.weight[:hidden] = 100 * torch.eye(hidden)
        model.vlm.lm_head.weight.zero_()
        model.vlm.lm_head.weight[1 : hidden + 1] = torch.eye(hidden)
    return model


def _image_inputs(last_token=5, seed=0, grids=GRIDS):
    """A one-sample image prompt whose greedy CoT is ``last_token + 1 ... ANCHOR``."""
    generator = torch.Generator().manual_seed(seed)
    ids = [1]
    for _, height, width in grids:
        ids += [11] + [10] * (height * width // 4) + [12]
    ids += [3, 4, last_token]
    input_ids = torch.tensor([ids])
    patches = sum(t * h * w for t, h, w in grids)
    xyz = torch.zeros(1, 1, 20, 3)
    xyz[0, 0, :, 0] = torch.arange(20) * 0.1
    return {
        "tokenized_data": {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": input_ids.eq(10).long(),
            "pixel_values": torch.randn(patches, PATCH_DIM, generator=generator),
            "image_grid_thw": torch.tensor(grids),
        },
        "ego_history_xyz": xyz,
        "ego_history_rot": torch.eye(3).expand(1, 1, 20, 3, 3).clone(),
    }


def _sample(model, inputs, num_traj_sets=1, max_generation_length=8, seed=3, top_k=1):
    device = model.device
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=model.dtype, enabled=model.dtype != torch.float32),
    ):
        return model.sample_trajectories_from_data(
            inputs,
            top_k=top_k,
            num_traj_sets=num_traj_sets,
            max_generation_length=max_generation_length,
            diffusion_kwargs={
                "inference_step": 3,
                "generator": torch.Generator(device=device).manual_seed(seed),
            },
            return_extra=True,
        )


def _sample_with_prefix_cache(model, inputs, **kwargs):
    """``_sample`` plus the prompt and CoT keys/values the expert attends, per layer."""
    installed = vars(model).get("_sample_actions")
    sample_actions = model._sample_actions
    columns = []

    def capture(cache, prefix_length, *args):
        columns.extend(
            torch.stack([layer.keys[:, :, :prefix_length], layer.values[:, :, :prefix_length]])
            for layer in cache.layers
        )
        return sample_actions(cache, prefix_length, *args)

    model._sample_actions = capture
    try:
        return _sample(model, inputs, **kwargs), torch.stack(columns)
    finally:
        if installed is None:
            del model._sample_actions
        else:
            model._sample_actions = installed


def _cpu_config(**kwargs):
    return AccelConfig(compile=False, cuda_graphs=False, **kwargs)


def test_kernel_selection_per_gpu(monkeypatch):
    def fake(capability, importable):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
        monkeypatch.setattr(kernels_module, "_importable", lambda name: name in importable)

    fake((9, 0), {"flash_attn", "flash_attn_interface"})
    assert resolve_kernels("auto", "cuda") == {"prefill": "fa3", "decode": "fa2", "expert": "fa2"}
    assert resolve_kernels("sdpa", "cuda")["prefill"] == "sdpa"
    fake((9, 0), {"flash_attn"})
    assert resolve_kernels("auto", "cuda")["prefill"] == "sdpa"
    with pytest.raises(ValueError, match="Hopper"):
        resolve_kernels("fa3", "cuda")
    fake((12, 1), {"flash_attn", "flash_attn_interface"})
    assert resolve_kernels("auto", "cuda") == {"prefill": "sdpa", "decode": "fa2", "expert": "fa2"}
    fake((8, 0), set())
    assert resolve_kernels("auto", "cuda")["decode"] is None
    assert resolve_kernels("auto", "cpu")["decode"] == "reference"
    with pytest.raises(ValueError, match="attention"):
        resolve_kernels("fa2", "cuda")


def test_parts_without_flash_attention_are_skipped(monkeypatch):
    model = _tiny_model(monkeypatch)
    monkeypatch.setattr(
        "alpamayo2_edge.accel.resolve_kernels",
        lambda attention, device: {"prefill": "sdpa", "decode": None, "expert": None},
    )
    with pytest.warns(RuntimeWarning, match="flash_attn") as record:
        handle = enable_acceleration(model, _cpu_config())
    assert "accelerated parts: none" in str(record[0].message)
    report = handle.describe()
    assert report["parts"] == []
    assert set(report["skipped"]) == {"prefill", "decode", "expert"}
    handle.remove()
    assert "_generate_with_shared_prefill" not in vars(model)


def test_config_validation_and_recompile_limit(monkeypatch):
    with pytest.raises(ValueError, match="warm-up"):
        AccelConfig(warmup=0)
    AccelConfig(warmup=0, cuda_graphs=False)
    with pytest.raises(ValueError, match="mode"):
        AccelConfig(mode="fast")
    dynamo_config = torch._dynamo.config
    monkeypatch.setattr(dynamo_config, "recompile_limit", 8)
    first, second = _tiny_model(monkeypatch), _tiny_model(monkeypatch)
    handles = [enable_acceleration(model, _cpu_config()) for model in (first, second)]
    assert dynamo_config.recompile_limit == 64
    handles[0].remove()
    handles[0].remove()  # A second removal does nothing.
    assert dynamo_config.recompile_limit == 64  # The other model still needs it.
    handles[1].remove()
    assert dynamo_config.recompile_limit == 8


def test_mask_columns_matches_mask_token_ids_processor():
    from alpamayo2_edge.models.alpamayo2_edge import MaskTokenIdsLogitsProcessor

    scores = torch.randn(3, 20)
    expected = MaskTokenIdsLogitsProcessor([2, 7, 19])(None, scores.clone())
    actual = MaskColumns([2, 7, 19], scores.device)(None, scores.clone())
    torch.testing.assert_close(actual, expected)


def test_expert_mask_detection():
    assert not _mask_hides_tokens(torch.zeros(2, 1, 4, 9))
    assert _mask_hides_tokens(torch.zeros(2, 1, 4, 9).index_fill_(3, torch.tensor([1]), -1e30))
    assert not _mask_hides_tokens(torch.ones(2, 9, dtype=torch.long))
    assert _mask_hides_tokens(torch.tensor([[1, 1, 0]]))
    assert _mask_hides_tokens(torch.tensor([[True, False]]))


def test_flash_cache_bookkeeping():
    cache = FlashKVCache(2, rows=2, max_len=16, n_kv_heads=2, head_dim=4, dtype=torch.float32,
                         device=torch.device("cpu"))  # fmt: skip
    cache.start(6, torch.tensor([3, 4]))
    assert cache.get_seq_length() == 6 and cache.get_mask_sizes(2, 0) == (8, 0)
    assert cache.layers[0].keys.shape == (2, 2, 6, 4)
    assert not cache.is_compileable
    with pytest.raises(RuntimeError, match="decode graphs"):
        cache.update(torch.zeros(2, 2, 1, 4), torch.zeros(2, 2, 1, 4), 0)
    cache.frozen = True
    keys = torch.randn(2, 2, 3, 4)
    for layer in range(2):
        out_keys, _ = cache.update(keys, keys, layer)
        torch.testing.assert_close(out_keys[:, :, 6:], keys)
        assert cache.get_seq_length() == (6 if layer == 0 else 9)
    cache.crop(7)  # Legacy positive crop: the new absolute length.
    assert cache.get_seq_length() == 7
    cache.crop(10)
    assert cache.get_seq_length() == 7
    cache.crop(-2)
    assert cache.get_seq_length() == 5
    with pytest.raises(RuntimeError, match="rows"):
        cache.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)
    with pytest.raises(NotImplementedError):
        cache.batch_repeat_interleave(2)


def test_eager_expert_reads_the_flash_cache_like_a_dynamic_cache(monkeypatch):
    from transformers.cache_utils import DynamicCache

    from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask

    model = _tiny_model(monkeypatch)
    config = model.vlm.config.text_config
    rows, length, cpu = 2, 7, torch.device("cpu")
    generator = torch.Generator().manual_seed(9)
    shape = (rows, config.num_key_value_heads, length, config.head_dim)
    dynamic = DynamicCache(config=config)
    for layer in range(config.num_hidden_layers):
        dynamic.update(
            torch.randn(shape, generator=generator), torch.randn(shape, generator=generator), layer
        )
    flash = FlashKVCache(
        config.num_hidden_layers, rows, 16, config.num_key_value_heads, config.head_dim,
        torch.float32, cpu,
    )  # fmt: skip
    flash.load(dynamic, torch.zeros(rows, 1, dtype=torch.long), 1)
    flash.frozen = True
    # CoTs of different lengths: the mask hides the shorter row's later columns.
    positions, mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([5, 7]),
        rope_deltas=torch.zeros(rows, 1, dtype=torch.long),
        kv_cache_seq_len=length,
        n_diffusion_tokens=4,
        b_star=rows,
        device=cpu,
    )

    def sample(cache):
        kwargs = {"inference_step": 2, "generator": torch.Generator().manual_seed(1)}
        with torch.inference_mode():
            actions = model._sample_actions(cache, length, positions, mask, rows, cpu, kwargs)
        assert cache.get_seq_length() == length
        return actions

    expected = sample(dynamic)
    handle = enable_acceleration(model, _cpu_config())
    torch.testing.assert_close(sample(flash), expected, rtol=1e-6, atol=1e-6)
    report = handle.describe()
    assert report["fallbacks"] == {"expert_mask": 1}
    # The graph for this shape is built anyway, ready for CoTs of equal length.
    assert [build["part"] for build in report["builds"]] == ["expert"]


@pytest.mark.parametrize("causal", [False, True])
def test_kvcache_attention_reference_appends_and_attends(causal):
    generator = torch.Generator().manual_seed(4)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k_cache = torch.randn(2, 10, 2, 8, generator=generator)
    v_cache = torch.randn(2, 10, 2, 8, generator=generator)
    k, v = torch.randn(2, 3, 2, 8, generator=generator), torch.randn(2, 3, 2, 8)
    seqlens = torch.tensor([4, 6], dtype=torch.int32)
    out = kernels_module.kvcache_attention(
        q, k_cache.clone(), v_cache.clone(), k, v, seqlens, 0.3, causal
    )
    for row, start in enumerate([4, 6]):
        keys = torch.cat([k_cache[row, :start], k[row]]).repeat_interleave(2, dim=1)
        values = torch.cat([v_cache[row, :start], v[row]]).repeat_interleave(2, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[row], keys) * 0.3
        if causal:
            visible = torch.arange(start + 3)[None, :] <= start + torch.arange(3)[:, None]
            scores = scores.masked_fill(~visible, float("-inf"))
        expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
        torch.testing.assert_close(out[row], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block", ["prefill_sdpa", "prefill_fa3", "decode"])
def test_fused_blocks_match_the_decoder_layer(monkeypatch, block):
    from transformers.cache_utils import DynamicCache

    from alpamayo2_edge.accel.decode import decoder_block, prefill_block

    text = _tiny_model(monkeypatch).vlm.model.language_model
    layer, config = text.layers[0], text.config
    rows, seq = 2, 6
    hidden = torch.randn(rows, seq, config.hidden_size, generator=torch.Generator().manual_seed(2))
    # Distinct temporal, height and width planes, as for vision tokens.
    positions = torch.arange(seq).view(1, 1, seq) + torch.tensor([0, 2, 5]).view(3, 1, 1)
    cos, sin = text.rotary_emb(hidden, positions.expand(3, rows, seq))
    shape = (rows, 8, config.num_key_value_heads, config.head_dim)
    with torch.inference_mode():
        cache = DynamicCache(config=config)
        expected = layer(
            hidden, position_embeddings=(cos, sin), past_key_values=cache, use_cache=True
        )
        k_cache, v_cache = torch.zeros(shape), torch.zeros(shape)
        if block == "decode":
            seqlens = torch.zeros(rows, dtype=torch.int32)
            actual = decoder_block(layer, hidden, cos, sin, k_cache, v_cache, seqlens, True)
        else:
            fa3 = block == "prefill_fa3"
            actual = prefill_block(layer, hidden, cos, sin, k_cache, v_cache, False, fa3)
    # Residual updates: the residual stream itself would hide small errors.
    torch.testing.assert_close(actual - hidden, expected - hidden)
    torch.testing.assert_close(k_cache[:, :seq].transpose(1, 2), cache.layers[0].keys)
    torch.testing.assert_close(v_cache[:, :seq].transpose(1, 2), cache.layers[0].values)


@pytest.mark.parametrize("num_traj_sets", [1, 2])
def test_accelerated_sampling_matches_eager_on_cpu(monkeypatch, num_traj_sets):
    model = _tiny_model(monkeypatch)
    inputs = _image_inputs()
    expected, expected_cache = _sample_with_prefix_cache(model, inputs, num_traj_sets=num_traj_sets)
    handle = enable_acceleration(model, _cpu_config())
    actual, actual_cache = _sample_with_prefix_cache(model, inputs, num_traj_sets=num_traj_sets)
    report = handle.describe()
    assert report["parts"] == ["prefill", "decode", "expert"]
    if num_traj_sets == 1:
        assert report["fallbacks"] == {}
    else:
        # Several CoTs share an eager prefill copied into the flash cache.
        assert report["fallbacks"] == {"prefill_unsupported": 1}
    generated = expected[3]["generated_ids"]
    assert generated.tolist() == [[6, 7, 8, ANCHOR, -1]] * num_traj_sets
    torch.testing.assert_close(actual[3]["generated_ids"], generated)
    # Decode positions barely move the trajectories, but they set the CoT keys.
    torch.testing.assert_close(actual_cache, expected_cache, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)
    handle.remove()
    for owner, name in (
        (model, "_generate_with_shared_prefill"),
        (model, "_sample_actions"),
        (model.vlm, "forward"),
    ):
        assert name not in vars(owner)


def test_warmup_builds_graphs_without_consuming_random_state(monkeypatch):
    model = _tiny_model(monkeypatch)
    handle = enable_acceleration(model, _cpu_config())
    state = torch.get_rng_state()
    handle.warmup(
        _image_inputs(), top_k=1, max_generation_length=8, diffusion_kwargs={"inference_step": 3}
    )
    assert torch.equal(torch.get_rng_state(), state)
    builds = [build["part"] for build in handle.describe()["builds"]]
    assert sorted(builds) == ["decode_sample", "expert", "prefill"]


def test_generation_limit_on_anchor_completes_the_flash_cache(monkeypatch):
    model = _tiny_model(monkeypatch)
    inputs = _image_inputs()
    # The limit lands on the anchor: the model forwards it after generation.
    expected, expected_cache = _sample_with_prefix_cache(model, inputs, max_generation_length=4)
    handle = enable_acceleration(model, _cpu_config())
    actual, actual_cache = _sample_with_prefix_cache(model, inputs, max_generation_length=4)
    assert expected[3]["generated_ids"].tolist() == [[6, 7, 8, ANCHOR]]
    torch.testing.assert_close(actual[3]["generated_ids"], expected[3]["generated_ids"])
    torch.testing.assert_close(actual_cache, expected_cache, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert handle.describe()["fallbacks"] == {}


def test_padded_prompts_fall_back(monkeypatch):
    model = _tiny_model(monkeypatch)
    inputs = _image_inputs()
    data = inputs["tokenized_data"]
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        data[key] = torch.cat([data[key].new_zeros(1, 1), data[key]], dim=1)
    expected = _sample(model, inputs)
    handle = enable_acceleration(model, _cpu_config())
    actual = _sample(model, inputs)
    torch.testing.assert_close(actual[0], expected[0])
    assert handle.describe()["fallbacks"] == {"decode_padding": 1, "expert_without_flash_cache": 1}


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def _gpu_model(monkeypatch):
    if resolve_kernels("sdpa", "cuda")["decode"] is None:
        pytest.skip("requires an importable flash_attn (FlashAttention 2)")
    return _tiny_model(monkeypatch, dtype="bfloat16").to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_compiled_cuda_graphs_match_eager_on_gpu(monkeypatch):
    model = _gpu_model(monkeypatch)
    inputs = _to_cuda(_image_inputs())
    expected = _sample(model, inputs)
    handle = enable_acceleration(model, AccelConfig(attention="sdpa"))
    actual = _sample(model, inputs)
    builds = handle.describe()["graph_builds"]
    again = _sample(model, inputs)
    report = handle.describe()
    assert report["fallbacks"] == {}
    assert report["graph_builds"] == builds  # Same shapes: replays only.
    assert {build["part"] for build in report["builds"]} == {"prefill", "decode_sample", "expert"}
    torch.testing.assert_close(actual[3]["generated_ids"], expected[3]["generated_ids"])
    torch.testing.assert_close(again[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.05, atol=0.05)
    handle.remove()


@requires_cuda
def test_graph_replays_read_new_inputs_on_gpu(monkeypatch):
    model = _gpu_model(monkeypatch)
    first = _to_cuda(_image_inputs())
    # Same shapes as the first input; new pixels, a longer CoT and new diffusion noise.
    second = _to_cuda(_image_inputs(last_token=4, seed=1))
    expected = _sample(model, second, seed=11)
    config = AccelConfig(mode="compile", attention="sdpa")  # No autotuning: same kernels.
    handle = enable_acceleration(model, config)
    _sample(model, first)  # Builds every graph on the first input.
    builds = handle.describe()["graph_builds"]
    replayed = _sample(model, second, seed=11)
    report = handle.describe()
    assert report["graph_builds"] == builds  # The second input only replays graphs.
    assert report["fallbacks"] == {}
    handle.remove()
    handle = enable_acceleration(model, config)
    direct = _sample(model, second, seed=11)  # Graphs built on the second input itself.
    handle.remove()
    # A replay must read every new input; stale static buffers would show up here.
    assert torch.equal(replayed[3]["generated_ids"], direct[3]["generated_ids"])
    assert torch.equal(replayed[0], direct[0])
    assert expected[3]["generated_ids"][0, :5].tolist() == [5, 6, 7, 8, ANCHOR]
    assert torch.equal(replayed[3]["generated_ids"], expected[3]["generated_ids"])
    torch.testing.assert_close(replayed[0], expected[0], rtol=0.05, atol=0.05)


def _generate(model, inputs, n_samples, seed):
    """``Alpamayo2Edge._generate_with_shared_prefill`` with the CLI's sampling settings."""
    from transformers import LogitsProcessorList, StoppingCriteriaList

    from alpamayo2_edge.models.alpamayo2_edge import (
        MaskDiscreteTrajectoryLogitsProcessor,
        MaskTokenIdsLogitsProcessor,
    )
    from alpamayo2_edge.models.expert_utils import StopAfterEOS

    config = copy.deepcopy(model.vlm.generation_config)
    config.update(
        do_sample=True,
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_beams=1,
        max_new_tokens=8,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    processors = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(100, 10),
            MaskDiscreteTrajectoryLogitsProcessor(110, 10),
            MaskTokenIdsLogitsProcessor([2]),
        ]
    )
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model._generate_with_shared_prefill(
            dict(inputs["tokenized_data"]),
            generation_config=config,
            n_samples_total=n_samples,
            logits_processor=processors,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(ANCHOR)]),
        )
    return outputs.sequences


@requires_cuda
@pytest.mark.parametrize("n_samples", [1, 3])
def test_graphed_sampler_draws_the_eager_random_numbers_on_gpu(monkeypatch, n_samples):
    model = _gpu_model(monkeypatch)
    with torch.no_grad():
        model.vlm.lm_head.weight.zero_()  # Uniform logits: every token is a random draw.
    inputs = _to_cuda(_image_inputs())
    handle = enable_acceleration(model, AccelConfig(attention="sdpa"))
    # The first call also builds the graphs, which must leave the RNG state unchanged.
    accelerated = [_generate(model, inputs, n_samples, seed) for seed in (0, 1)]
    report = handle.describe()
    assert "decode_sample" in {build["part"] for build in report["builds"]}
    assert "sampler_unsupported" not in report["fallbacks"]
    handle.remove()
    eager = [_generate(model, inputs, n_samples, seed) for seed in (0, 1)]
    assert not torch.equal(eager[0], eager[1])
    for got, want in zip(accelerated, eager, strict=True):
        assert torch.equal(got, want)


def _to_cuda(inputs):
    from alpamayo2_edge.helper import to_device

    return to_device(inputs, device="cuda")
