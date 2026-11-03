"""Numerical CPU regressions for Cosmos3-Edge expert attention and cache reuse."""

import copy
import math

import pytest
import torch
from transformers import Cosmos3EdgeTextConfig, Cosmos3EdgeTextModel

from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask


def _text_config(attention_backend="sdpa"):
    config = Cosmos3EdgeTextConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "mrope_section": [2, 3, 3]},
        attention_dropout=0.0,
    )
    config._attn_implementation = attention_backend
    return config


@pytest.fixture(params=["eager", "sdpa"])
def expert_backbone(request):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = Cosmos3EdgeTextModel(_text_config(request.param)).eval()
    return model


def _expert_inputs(prefix_length=5, offset=3):
    generator = torch.Generator().manual_seed(17)
    embeddings = torch.randn(1, 3, 64, generator=generator)
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([offset]),
        rope_deltas=torch.tensor([[-1]]),
        kv_cache_seq_len=prefix_length,
        n_diffusion_tokens=3,
        b_star=1,
        device=torch.device("cpu"),
        prefix_mask=torch.tensor([[0, 1, 1, 1, 1]])[:, :prefix_length],
    )
    return embeddings, position_ids, attention_mask


@torch.no_grad()
def _prefix_cache(model):
    # The VLM and expert have different hidden widths but share KV head geometry.
    prefix_config = copy.deepcopy(model.config)
    prefix_config.hidden_size = 128
    prefix_config.intermediate_size = 192
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        prefix_model = Cosmos3EdgeTextModel(prefix_config).eval()
    generator = torch.Generator().manual_seed(23)
    return prefix_model(
        inputs_embeds=torch.randn(1, 5, 128, generator=generator),
        use_cache=True,
    ).past_key_values


@torch.no_grad()
def _run_expert(model, cache, embeddings, position_ids, attention_mask):
    return model(
        inputs_embeds=embeddings,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        is_causal=False,
    ).last_hidden_state


def test_expert_mask_preserves_anchor_positions_padding_and_all_action_connections():
    position_ids, mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([3, 5]),
        rope_deltas=torch.tensor([[-2], [-1]]),
        kv_cache_seq_len=5,
        n_diffusion_tokens=3,
        b_star=2,
        device=torch.device("cpu"),
        prefix_mask=torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
    )
    torch.testing.assert_close(
        position_ids,
        torch.tensor([[[1, 2, 3], [4, 5, 6]]]).expand(3, -1, -1),
    )
    assert mask.shape == (2, 1, 3, 8)
    assert mask.dtype == torch.float32
    allowed = torch.tensor(
        [
            [False, True, True, False, False, True, True, True],
            [False, False, True, True, True, True, True, True],
        ]
    )[:, None, None, :].expand_as(mask)
    torch.testing.assert_close(mask == 0, allowed)
    assert torch.all(mask[~allowed] == torch.finfo(mask.dtype).min)


def test_later_action_changes_first_action_only_with_noncausal_mask(expert_backbone):
    """A causal control guards against a test that merely observes changed inputs."""
    cache = _prefix_cache(expert_backbone)
    embeddings, position_ids, mask = _expert_inputs()
    changed_embeddings = embeddings.clone()
    changed_embeddings[:, -1] = -3 * changed_embeddings[:, -1]

    baseline = _run_expert(expert_backbone, copy.deepcopy(cache), embeddings, position_ids, mask)
    changed = _run_expert(
        expert_backbone, copy.deepcopy(cache), changed_embeddings, position_ids, mask
    )
    assert torch.isfinite(baseline).all()
    assert (baseline[:, 0] - changed[:, 0]).abs().max().item() > 1e-4

    causal_mask = mask.clone()
    causal_mask[..., 5:] = torch.triu(torch.full((3, 3), torch.finfo(mask.dtype).min), diagonal=1)
    causal_baseline = _run_expert(
        expert_backbone, copy.deepcopy(cache), embeddings, position_ids, causal_mask
    )
    causal_changed = _run_expert(
        expert_backbone, copy.deepcopy(cache), changed_embeddings, position_ids, causal_mask
    )
    torch.testing.assert_close(causal_baseline[:, 0], causal_changed[:, 0], rtol=0, atol=0)


def test_masked_cache_values_are_ignored_and_prefix_survives_denoising(expert_backbone):
    cache = _prefix_cache(expert_backbone)
    saved_prefix = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    embeddings, position_ids, mask = _expert_inputs()
    baseline = _run_expert(expert_backbone, cache, embeddings, position_ids, mask)

    assert cache.get_seq_length() == 8
    for layer, (keys, values) in zip(cache.layers, saved_prefix, strict=True):
        torch.testing.assert_close(layer.keys[..., :5, :], keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values[..., :5, :], values, rtol=0, atol=0)
    cache.crop(5)
    assert cache.get_seq_length() == 5

    poisoned_cache = copy.deepcopy(cache)
    for layer in poisoned_cache.layers:
        # Zero is padding; three and four follow this row's trajectory anchor.
        layer.keys[..., [0, 3, 4], :] += 100
        layer.values[..., [0, 3, 4], :] -= 100
    poisoned = _run_expert(expert_backbone, poisoned_cache, embeddings, position_ids, mask)
    torch.testing.assert_close(poisoned, baseline, rtol=0, atol=0)

    visible_cache = copy.deepcopy(cache)
    for layer in visible_cache.layers:
        layer.values[..., 1, :] += 100
    visible = _run_expert(expert_backbone, visible_cache, embeddings, position_ids, mask)
    assert (visible - baseline).abs().max().item() > 1e-4

    repeated = _run_expert(expert_backbone, cache, embeddings, position_ids, mask)
    torch.testing.assert_close(repeated, baseline, rtol=0, atol=0)
    cache.crop(5)
    for layer, (keys, values) in zip(cache.layers, saved_prefix, strict=True):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)


def _expert_config():
    from alpamayo2_edge.models.expert import ExpertModelConfig

    return ExpertModelConfig(
        llm_config=_text_config(),
        action_space_cfg={
            "_target_": "alpamayo2_edge.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace",
            "n_waypoints": 3,
        },
        diffusion_cfg={"_target_": "alpamayo2_edge.diffusion.flow_matching.FlowMatching"},
        action_in_proj_cfg={
            "_target_": "alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2",
            "num_enc_layers": 1,
            "hidden_size": 32,
            "num_fourier_feats": 4,
        },
        action_out_proj_cfg={"_target_": "torch.nn.Linear"},
        expert_non_causal_attention=True,
    )


def test_expert_state_dict_preserves_attention_and_action_projection_geometry():
    from alpamayo2_edge.models.expert import ExpertModel

    config = _expert_config()
    model = ExpertModel(config).eval()
    state = model.state_dict()
    expected_shapes = {
        "expert.layers.0.self_attn.q_proj.weight": (64, 64),
        "expert.layers.0.self_attn.k_proj.weight": (32, 64),
        "expert.layers.0.self_attn.v_proj.weight": (32, 64),
        "expert.layers.0.self_attn.o_proj.weight": (64, 64),
        "expert.layers.1.mlp.fc2.weight": (64, 96),
        "action_in_proj.encoder.trunk.0.weight": (32, 12),
        "action_in_proj.encoder.trunk.3.weight": (64, 32),
        "action_out_proj.weight": (2, 64),
        "action_out_proj.bias": (2,),
    }
    for name, shape in expected_shapes.items():
        assert state[name].shape == shape, name
    assert not any("embed_tokens" in name or "lm_head" in name for name in state)
    assert model.action_space.get_action_space_dims() == (3, 2)
    restored = ExpertModel(config)
    incompatible = restored.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize(
    "action_space",
    ["UnicycleAccelCurvatureActionSpace", "UnicycleAccelCurvatureCourseActionSpace"],
)
def test_expert_pretrained_roundtrip_reconstructs_constants_and_preserves_weights(
    tmp_path, action_space
):
    from safetensors import safe_open

    from alpamayo2_edge.models.expert import ExpertModel

    config = _expert_config()
    config.action_space_cfg["_target_"] = f"alpamayo2_edge.action_space.{action_space}"
    normalization = {
        "accel_mean": 1.25,
        "accel_std": 2.5,
        "curvature_mean": -0.125,
        "curvature_std": 0.375,
    }
    config.action_space_cfg.update(normalization)
    config.action_in_proj_cfg.update(max_freq=37.0, num_fourier_feats=8)
    original = ExpertModel(config).eval()
    original.save_pretrained(tmp_path)
    restored, loading_info = ExpertModel.from_pretrained(
        tmp_path, local_files_only=True, output_loading_info=True
    )
    assert not any(loading_info.values()), loading_info
    assert type(restored.action_space).__name__ == action_space

    original_parameters = dict(original.named_parameters())
    restored_parameters = dict(restored.named_parameters())
    assert original_parameters.keys() == restored_parameters.keys()
    for name, value in original_parameters.items():
        torch.testing.assert_close(restored_parameters[name], value, rtol=0, atol=0, msg=name)

    original_buffers = dict(original.named_buffers())
    restored_buffers = dict(restored.named_buffers())
    frequency_names = [name for name in original_buffers if name.endswith(".freqs")]
    assert len(frequency_names) == 3
    expected_frequencies = torch.logspace(0, math.log10(37.0), 4)[None, :]
    for name in frequency_names:
        torch.testing.assert_close(original_buffers[name], expected_frequencies, rtol=0, atol=0)
        torch.testing.assert_close(restored_buffers[name], original_buffers[name], rtol=0, atol=0)
    for name, value in normalization.items():
        expected_value = torch.tensor(value)
        torch.testing.assert_close(
            getattr(original.action_space, name), expected_value, rtol=0, atol=0
        )
        torch.testing.assert_close(
            getattr(restored.action_space, name), expected_value, rtol=0, atol=0
        )

    # The checkpoint deliberately omits these constants, so equality above must
    # come from config reconstruction through HF's meta-device loading path.
    constant_names = set(frequency_names) | {f"action_space.{name}" for name in normalization}
    with safe_open(tmp_path / "model.safetensors", framework="pt") as checkpoint:
        assert constant_names.isdisjoint(checkpoint.keys())

    actions = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]])
    timesteps = torch.tensor([[[0.25]]])
    with torch.no_grad():
        torch.testing.assert_close(
            restored.action_in_proj(actions, timesteps),
            original.action_in_proj(actions, timesteps),
            rtol=0,
            atol=0,
        )
