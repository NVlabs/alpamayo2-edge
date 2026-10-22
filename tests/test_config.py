import copy
import json
import os
from pathlib import Path

import hydra.utils as hyu
import pytest
import torch

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


# Trajectory action space of the course-heading Edge checkpoint, as saved by training.
_COURSE_ACTION_SPACE = {
    "_target_": "alpamayo.action_space.UnicycleAccelCurvatureCourseActionSpace",
    "a_lambda": 0.0001,
    "a_ridge": 0.0001,
    "accel_bounds": [-9.8, 9.8],
    "accel_mean": 0.02902694707164455,
    "accel_std": 0.6810426736454882,
    "curvature_bounds": [-0.33, 0.33],
    "curvature_mean": 0.0002692167976330542,
    "curvature_std": 0.026148280660833106,
    "dt": 0.1,
    "kappa_lambda": 0.0001,
    "kappa_ridge": 0.0001,
    "n_waypoints": 64,
    "slip_gradient": 0.0025,
    "theta_lambda": 3e-06,
    "theta_ridge": 1e-08,
    "v_lambda": 1e-06,
    "v_ridge": 0.0001,
}


def _assert_space_matches_config(space, config):
    assert type(space) is hyu.get_class(config["_target_"])
    for name, saved in config.items():
        if name.startswith("_"):
            continue
        value = getattr(space, name)
        if isinstance(value, torch.Tensor):
            # Normalization buffers hold the saved value at float32 precision.
            value, saved = value.item(), torch.tensor(saved).item()
        if isinstance(value, tuple):
            value = list(value)
        assert value == saved, name


def test_course_action_space_normalizes_in_tokenizer_and_expert():
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace

    raw = {
        "future_traj_tokenizer_cfg": {
            "_recursive_": False,
            "_target_": "alpamayo.action_space.discrete_action_space.DiscreteTrajectoryTokenizer",
            "action_space_cfg": copy.deepcopy(_COURSE_ACTION_SPACE),
            "dims_max": [10, 10],
            "dims_min": [-10, -10],
            "num_bins": 3000,
        },
        "expert_config": {"action_space_cfg": copy.deepcopy(_COURSE_ACTION_SPACE)},
    }
    normalized = normalize_checkpoint_config(raw)
    tokenizer_space_cfg = normalized["future_traj_tokenizer_cfg"]["action_space_cfg"]
    expert_space_cfg = normalized["expert_config"]["action_space_cfg"]
    for config in (tokenizer_space_cfg, expert_space_cfg):
        assert config == {
            **_COURSE_ACTION_SPACE,
            "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
        }
    tokenizer = hyu.instantiate(normalized["future_traj_tokenizer_cfg"])
    assert isinstance(tokenizer.action_space, UnicycleAccelCurvatureCourseActionSpace)
    _assert_space_matches_config(tokenizer.action_space, tokenizer_space_cfg)
    _assert_space_matches_config(hyu.instantiate(expert_space_cfg), expert_space_cfg)


@pytest.mark.skipif(
    "ALPAMAYO2_EDGE_CHECKPOINT" not in os.environ,
    reason="set ALPAMAYO2_EDGE_CHECKPOINT to a local checkpoint directory",
)
def test_local_checkpoint_action_spaces_keep_saved_parameters():
    path = Path(os.environ["ALPAMAYO2_EDGE_CHECKPOINT"])
    config = Alpamayo2EdgeConfig(
        **normalize_checkpoint_config(json.loads((path / "config.json").read_text()))
    )
    tokenizer = hyu.instantiate(config.future_traj_tokenizer_cfg)
    _assert_space_matches_config(
        tokenizer.action_space, config.future_traj_tokenizer_cfg["action_space_cfg"]
    )
    expert_space_cfg = config.expert_config.action_space_cfg
    _assert_space_matches_config(hyu.instantiate(expert_space_cfg), expert_space_cfg)
