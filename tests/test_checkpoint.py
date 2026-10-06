import json
from pathlib import Path

import pytest

from alpamayo2_edge.checkpoint import resolve_checkpoint, validate_checkpoint


def model_layout(path: Path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    for name in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
        (path / name).write_text("{}")
    (path / "chat_template.jinja").write_text("template")
    (path / "model.safetensors").touch()
    return path


def test_resolves_model_and_bundle_without_network(tmp_path):
    model = model_layout(tmp_path / "bundle" / "model")
    assert resolve_checkpoint(str(model)) == model
    assert resolve_checkpoint(str(model.parent)) == model


def test_missing_local_path_fails_as_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "missing"))


def test_missing_weight_shard_is_reported_before_loading(tmp_path):
    model = model_layout(tmp_path / "model")
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"vlm.weight": "missing.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        validate_checkpoint(model)


def test_loading_incomplete_processor_assets_fails(tmp_path):
    model = model_layout(tmp_path / "model")
    (model / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        validate_checkpoint(model)
