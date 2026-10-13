import json
from pathlib import Path

import pytest

from alpamayo2_edge.checkpoint import load_model, resolve_checkpoint, validate_checkpoint


def model_layout(path: Path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    for name in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
        (path / name).write_text("{}")
    (path / "chat_template.jinja").write_text("template")
    (path / "model.safetensors").touch()
    return path


def test_resolves_model_directory_and_its_parent_without_network(tmp_path):
    model = model_layout(tmp_path / "checkpoint" / "model")
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


@pytest.mark.parametrize(
    "processor_config",
    [
        {},
        {"processor_class": "Qwen3VLProcessor"},
        {"image_processor": None},
        {"image_processor": {}},
        {"image_processor": "Qwen2VLImageProcessor"},
    ],
)
def test_unified_processor_without_image_configuration_is_incomplete(tmp_path, processor_config):
    model = model_layout(tmp_path / "model")
    (model / "preprocessor_config.json").unlink()
    (model / "processor_config.json").write_text(json.dumps(processor_config))
    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        validate_checkpoint(model)


def test_trainer_checkpoint_loads_with_unified_processor_configuration(tmp_path, monkeypatch):
    """HF5 Trainer exports must pass the same loader used by the inference CLI."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoProcessor,
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
        Trainer,
        TrainingArguments,
    )

    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokens = [
        "<pad>",
        "<unk>",
        "</s>",
        "<i0>",
        "<i1>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>")
        ),
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="</s>",
    )
    source = tmp_path / "tokenizer"
    tokenizer.save_pretrained(source)
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 784, "longest_edge": 3136}),
        video_processor=Qwen3VLVideoProcessor(),
        chat_template="{{ messages[0]['content'] }}",
    )
    config = Alpamayo2EdgeConfig(
        vlm_name_or_path=str(source),
        dtype="float32",
        enable_expert=False,
        history_vocab_size=1,
        future_vocab_size=1,
        vlm_config=Qwen3VLConfig(
            text_config={
                "vocab_size": 16,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "pad_token_id": 0,
                "eos_token_id": 2,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
                "deepstack_visual_indexes": [],
            },
        ),
    )
    model = Alpamayo2Edge(config)
    checkpoint = tmp_path / "checkpoint"
    Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path / "run"), use_cpu=True, report_to="none"),
        processing_class=processor,
    ).save_model(checkpoint)
    assert not (checkpoint / "preprocessor_config.json").exists()
    source.rename(tmp_path / "unused-original-tokenizer")

    loaded, info = load_model(str(checkpoint), device="cpu", dtype="float32", local_files_only=True)
    assert not any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    restored = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    assert restored.image_processor.size == {"shortest_edge": 784, "longest_edge": 3136}
    assert restored.chat_template == processor.chat_template
