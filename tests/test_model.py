"""CPU regression tests for CoT boundaries and trajectory sampling semantics."""

from types import SimpleNamespace

import pytest
import torch

from alpamayo2_edge.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
)


def test_missing_expert_anchor_is_an_error():
    with pytest.raises(ValueError, match="traj_future_start.*rows.*1"):
        find_eos_offset(torch.tensor([[4, 9, 2], [4, 3, 2]]), 9, torch.device("cpu"))


def test_expert_mask_hides_padding_and_post_anchor_cache_but_is_noncausal():
    positions, mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([3, 5]),
        rope_deltas=torch.tensor([[-1], [-2]]),
        kv_cache_seq_len=6,
        n_diffusion_tokens=3,
        b_star=2,
        device=torch.device("cpu"),
        prefix_mask=torch.tensor([[0, 1], [1, 1]]),
    )
    assert positions.shape == (3, 2, 3)
    torch.testing.assert_close(positions[0], torch.tensor([[2, 3, 4], [3, 4, 5]]))
    assert (mask[0, :, :, 0] < -1e20).all()
    assert (mask[0, :, :, 3:6] < -1e20).all()
    assert (mask[1, :, :, 5:6] < -1e20).all()
    assert (mask[:, :, :, 6:] == 0).all()
    assert (mask[0, :, :, 1:3] == 0).all()


def test_stopping_criterion_caches_anchor_before_stopping():
    stop = StopAfterEOS(9)
    assert not stop(torch.tensor([[1, 9], [1, 4]]), None)
    assert not stop(torch.tensor([[1, 9, 5], [1, 4, 9]]), None)
    assert stop(torch.tensor([[1, 9, 5, 8], [1, 4, 9, 8]]), None)


class _Cache:
    def __init__(self, length):
        self.length = length
        self.repeat_factor = None
        self.crops = []

    def get_seq_length(self):
        return self.length

    def batch_repeat_interleave(self, repeats):
        self.repeat_factor = repeats

    def crop(self, length):
        self.crops.append(length)
        self.length = length


def _sampling_model(monkeypatch, *, include_anchor=True, cache_length=5):
    from transformers import GenerationConfig

    import alpamayo2_edge.models.alpamayo2_edge as module
    from alpamayo2_edge.diffusion.flow_matching import FlowMatching

    cache = _Cache(cache_length)
    calls = []
    generations = []
    pending_tokens = []

    class VLM:
        generation_config = GenerationConfig(eos_token_id=2)
        model = SimpleNamespace(rope_deltas=torch.tensor([[-1], [-2]]))

        def generate(self, input_ids, generation_config, **kwargs):
            generations.append(generation_config)
            anchor = 9 if include_anchor else 8
            generated = torch.tensor([[3, anchor, 7, 7], [4, 5, anchor, 7]])
            return SimpleNamespace(
                sequences=torch.cat([input_ids.repeat_interleave(2, dim=0), generated], dim=1),
                past_key_values=cache,
            )

    def finish_cache(**kwargs):
        pending_tokens.append(kwargs)
        cache.length += kwargs["input_ids"].shape[-1]
        return SimpleNamespace(past_key_values=cache)

    vlm = VLM()
    vlm.model = SimpleNamespace(rope_deltas=VLM.model.rope_deltas, language_model=finish_cache)

    def denoiser(**kwargs):
        calls.append(kwargs)
        assert cache.length == 5
        cache.length += 4
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])

    model = module.Alpamayo2Edge.__new__(module.Alpamayo2Edge)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        enable_expert=True,
        tokens_per_future_traj=8,
        traj_ids={"history_id0": 100, "future_id0": 110, "future_start": 9},
        history_vocab_size=10,
        future_vocab_size=10,
        traj_vocab_size=20,
    )
    model.vlm = vlm
    model.tokenizer = SimpleNamespace(pad_token_id=0)
    model.history_traj_tokenizer = None
    model.future_traj_tokenizer = None
    model.expert = SimpleNamespace(
        action_space=SimpleNamespace(
            get_action_space_dims=lambda: (4, 2),
            action_to_traj=lambda action, xyz, rot: (
                torch.cat([action, action[..., :1]], dim=-1),
                torch.eye(3).expand(action.shape[0], 4, 3, 3),
            ),
        ),
        action_in_proj=lambda x, t: x,
        action_out_proj=lambda x: x,
        expert=SimpleNamespace(__call__=denoiser),
        config=SimpleNamespace(expert_non_causal_attention=True),
        diffusion=FlowMatching(x_dims=(4, 2)),
    )

    class Denoiser:
        config = SimpleNamespace(_attn_implementation="sdpa")

        def __call__(self, **kwargs):
            return denoiser(**kwargs)

    model.expert.expert = Denoiser()
    monkeypatch.setattr(module, "fuse_traj_tokens", lambda a, b, ids, data, config: ids)
    monkeypatch.setattr(module, "extract_text_tokens", lambda tokenizer, ids: {"cot": ["A", "B"]})
    data = {
        "tokenized_data": {
            "input_ids": torch.tensor([[1, 6]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        "ego_history_xyz": torch.zeros(1, 1, 3, 3),
        "ego_history_rot": torch.eye(3).expand(1, 1, 3, 3, 3),
    }
    return model, data, cache, calls, generations, pending_tokens


def test_samples_share_cot_cache_and_crop_each_diffusion_step(monkeypatch):
    model, data, cache, calls, generations, pending = _sampling_model(monkeypatch)
    xyz, rot, logprob, extra = model.sample_trajectories_from_data(
        data, num_traj_sets=2, num_traj_samples=3, return_extra=True
    )
    assert generations[0].num_return_sequences == 2
    assert cache.repeat_factor == 3
    assert len(calls) == 10
    assert cache.length == 5
    assert cache.crops[-10:] == [5] * 10
    assert calls[0]["inputs_embeds"].shape == (6, 4, 2)
    assert calls[0]["attention_mask"].shape == (6, 1, 4, 9)
    assert calls[0]["is_causal"] is False
    assert xyz.shape == (1, 2, 3, 4, 3)
    assert rot.shape == (1, 2, 3, 4, 3, 3)
    assert logprob.shape == (1, 2, 3, 4)
    assert extra["cot"].tolist() == [[["A", "A", "A"], ["B", "B", "B"]]]
    torch.testing.assert_close(
        extra["generated_ids"], torch.tensor([[3, 9, -1, -1], [4, 5, 9, -1]])
    )
    assert not pending


def test_incomplete_cot_never_reaches_expert(monkeypatch):
    model, data, cache, calls, generations, pending = _sampling_model(
        monkeypatch, include_anchor=False
    )
    with pytest.raises(ValueError, match="traj_future_start"):
        model.sample_trajectories_from_data(data, num_traj_sets=2)
    assert not calls


def test_generation_limit_on_anchor_finishes_only_uncached_token(monkeypatch):
    model, data, cache, calls, generations, pending = _sampling_model(monkeypatch, cache_length=4)
    model.sample_trajectories_from_data(data, num_traj_sets=2)
    assert len(pending) == 1
    torch.testing.assert_close(pending[0]["input_ids"], torch.tensor([[7], [9]]))
    assert pending[0]["past_key_values"] is cache
    assert pending[0]["attention_mask"].shape == (2, 5)
    assert len(calls) == 10


@pytest.mark.parametrize("cache_length", [4, 5])
def test_left_padding_is_removed_from_hf5_rotary_offsets(monkeypatch, cache_length):
    model, data, cache, calls, generations, pending = _sampling_model(
        monkeypatch, cache_length=cache_length
    )
    data["tokenized_data"]["input_ids"][0, 0] = 0
    data["tokenized_data"]["attention_mask"][0, 0] = 0
    model.sample_trajectories_from_data(data, num_traj_sets=2)
    # Absolute anchor offsets [4, 5] + HF5 deltas [-1, -2] - one pad.
    torch.testing.assert_close(calls[0]["position_ids"][0, :, 0], torch.tensor([2, 2]))
    assert (calls[0]["attention_mask"][:, :, :, 0] < -1e20).all()
    if cache_length == 4:
        # The uncached token at column 4 follows the same padded-position rule.
        torch.testing.assert_close(pending[0]["position_ids"][0, :, 0], torch.tensor([2, 1]))


def test_an_anchor_in_the_prompt_does_not_hide_incomplete_generation(monkeypatch):
    model, data, cache, calls, generations, pending = _sampling_model(
        monkeypatch, include_anchor=False
    )
    data["tokenized_data"]["input_ids"][0, 0] = 9
    with pytest.raises(ValueError, match="traj_future_start"):
        model.sample_trajectories_from_data(data, num_traj_sets=2)
    assert not calls


def test_legacy_packed_inputs_get_multimodal_position_types(monkeypatch):
    model, data, cache, calls, generations, pending = _sampling_model(monkeypatch)
    model.config.vlm_config = SimpleNamespace(image_token_id=6, video_token_id=8)
    data["tokenized_data"]["pixel_values"] = torch.zeros(1, 4)
    data["tokenized_data"]["image_grid_thw"] = torch.ones(1, 3, dtype=torch.long)
    generated_inputs = []
    original_generate = model.vlm.generate

    def capture_generate(**kwargs):
        generated_inputs.append(kwargs)
        return original_generate(**kwargs)

    monkeypatch.setattr(model.vlm, "generate", capture_generate)
    model.sample_trajectories_from_data(data, num_traj_sets=2)
    torch.testing.assert_close(generated_inputs[0]["mm_token_type_ids"], torch.tensor([[0, 1]]))


@pytest.mark.parametrize("generation_limit", [1, 8])
def test_real_qwen_generation_and_expert_integration_on_cpu(monkeypatch, generation_limit):
    """Exercise HF generate/cache/position APIs using small real model layers."""
    from transformers import Qwen3VLConfig

    import alpamayo2_edge.models.alpamayo2_edge as module
    from alpamayo2_edge.config import Alpamayo2EdgeConfig

    text_config = {
        "model_type": "qwen3_vl_text",
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
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_heads": 2,
                "out_hidden_size": 64,
                "deepstack_visual_indexes": [],
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
        traj_ids={"history_id0": 100, "future_id0": 110, "future_start": 9},
        history_vocab_size=10,
        future_vocab_size=10,
    )
    monkeypatch.setattr(
        module, "build_alpamayo2_edge_tokenizer", lambda *args: SimpleNamespace(pad_token_id=0)
    )
    monkeypatch.setattr(module, "fuse_traj_tokens", lambda a, b, ids, data, config: ids)
    model = module.Alpamayo2Edge(config).eval()

    class AnchorHead(torch.nn.Module):
        def forward(self, hidden):
            logits = hidden.new_full((*hidden.shape[:-1], 128), -1000)
            logits[..., 9] = 1000
            return logits

    model.vlm.lm_head = AnchorHead()
    xyz = torch.zeros(1, 1, 20, 3)
    xyz[0, 0, :, 0] = torch.arange(20) * 0.1
    data = {
        "tokenized_data": {"input_ids": torch.tensor([[1, 6]]), "attention_mask": torch.ones(1, 2)},
        "ego_history_xyz": xyz,
        "ego_history_rot": torch.eye(3).expand(1, 1, 20, 3, 3),
    }
    predicted, rotations, logprob = model.sample_trajectories_from_data(
        data, num_traj_sets=2, num_traj_samples=2, max_generation_length=generation_limit
    )
    assert predicted.shape == (1, 2, 2, 4, 3)
    assert rotations.shape == (1, 2, 2, 4, 3, 3)
    assert torch.isfinite(predicted).all()
    assert torch.isfinite(rotations).all()
    assert (logprob == 0).all()
