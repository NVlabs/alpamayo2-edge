import copy

import pytest

from alpamayo2_edge.config import Alpamayo2EdgeConfig, normalize_checkpoint_config


def test_normalizes_legacy_targets_without_changing_input_or_architecture():
    raw = {
        "model_type": "alpamayo3",
        "architectures": ["Alpamayo3"],
        "hist_traj_tokenizer_cfg": {
            "_target_": "alpamayo.tokenizer.trajectory.delta_tokenizer.DeltaTrajectoryTokenizer",
            "pad_origin_at_beginning": False,
        },
        "expert_config": {
            "model_type": "alpamayo1_expert",
            "action_in_proj_cfg": {
                "_target_": "reasoning_vla.submodules.action_in_proj.PerWaypointActionInProjV2"
            },
            "llm_config": {"model_type": "qwen3_vl_text", "hidden_size": 512},
        },
    }
    before = copy.deepcopy(raw)
    normalized = normalize_checkpoint_config(raw)
    assert raw == before
    assert normalized["model_type"] == "alpamayo2_edge"
    assert normalized["architectures"] == ["Alpamayo2Edge"]
    assert normalized["expert_config"]["llm_config"] == before["expert_config"]["llm_config"]
    assert (
        normalized["hist_traj_tokenizer_cfg"]["_target_"]
        == "alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer"
    )
    assert normalized["expert_config"]["model_type"] == "alpamayo2_edge_expert"
    assert normalize_checkpoint_config(normalized) == normalized


def test_unknown_hydra_target_is_rejected():
    with pytest.raises(ValueError, match="Unsupported.*target"):
        normalize_checkpoint_config({"hist_traj_tokenizer_cfg": {"_target_": "os.system"}})


def test_history_count_derived_from_tokenizer_without_padding_or_truncation():
    config = Alpamayo2EdgeConfig(
        tokens_per_history_traj=57, hist_traj_tokenizer_cfg={"pad_origin_at_beginning": False}
    )
    assert config.history_num_steps == 20
    assert config.frame_label == "relative_time"
    with pytest.raises(ValueError, match="divisible"):
        Alpamayo2EdgeConfig(tokens_per_history_traj=56)
    with pytest.raises(ValueError, match="history_num_steps"):
        Alpamayo2EdgeConfig(tokens_per_history_traj=57, history_num_steps=16)


def test_config_defaults_serialize_without_downloading_models():
    config = Alpamayo2EdgeConfig()
    assert config.to_dict()["model_type"] == "alpamayo2_edge"
    assert config.history_num_steps == 20
