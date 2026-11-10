"""Input contracts that protect camera, timestamp, and trajectory alignment."""

import os
from types import SimpleNamespace

import pytest
import torch

from alpamayo2_edge import helper
from alpamayo2_edge.chat_template import build_vqa_conversation
from alpamayo2_edge.helper import create_messages, to_device
from alpamayo2_edge.sample import (
    canonicalize_sample,
    canonicalize_vqa_sample,
    validate_sample,
    validate_vqa_sample,
)


def raw_sample():
    cameras = [0] * 3 + [1] * 5 + [2] * 3 + [3] * 3 + [5] * 3 + [6] * 5
    timestamps = []
    for camera in (0, 1, 2, 3, 5, 6):
        offsets = (
            [-2_000_000, -1_000_000, -500_000, -100_000, 0]
            if camera in (1, 6)
            else [-1_000_000, -100_000, 0]
        )
        timestamps.extend(1_700_000_000_000_000 + offset for offset in offsets)
    return {
        "image_frames": torch.zeros(22, 1, 3, 32, 64, dtype=torch.uint8),
        "camera_indices": torch.tensor(cameras, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "ego_history_xyz": torch.zeros(1, 20, 3),
        "ego_history_rot": torch.eye(3).expand(1, 20, 3, 3).contiguous(),
        "nav_text": ["Continue straight"],
    }


def raw_vqa_sample():
    cameras = [camera for camera in (0, 1, 2, 3, 5, 6) for _ in range(4)]
    offsets = [-300_000, -200_000, -100_000, 0] * 6
    return {
        "image_frames": torch.zeros(24, 1, 3, 32, 64, dtype=torch.uint8),
        "camera_indices": torch.tensor(cameras, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(offsets, dtype=torch.int64) + 1_700_000_000_000_000,
    }


def config():
    return SimpleNamespace(
        tokens_per_history_traj=57,
        history_num_steps=20,
        include_camera_ids=True,
        frame_label="relative_time",
        token_layout="camera_ts",
        encode_as_video=False,
    )


def test_canonicalize_sample_normalizes_camera_pixel_bounds():
    data = raw_sample()
    data["camera_pixel_bounds"] = {"1": [65536, 131072]}
    canonical = canonicalize_sample(data)
    assert canonical["camera_pixel_bounds"][1] == (65536, 131072)
    assert canonical["camera_pixel_bounds"][0] == (81920, 98304)
    assert data["camera_pixel_bounds"] == {"1": [65536, 131072]}


def test_prompt_preserves_camera_order_relative_timestamps_and_navigation():
    data = raw_sample()
    # Capture times need not coincide with the nominal sampling grid.
    data["absolute_timestamps"][0] -= 160_000
    messages = create_messages(data, config())
    assert [message["role"] for message in messages] == ["system", "user"]
    content = messages[1]["content"]
    text = "".join(entry["text"] if entry["type"] == "text" else "[IMAGE]" for entry in content)
    assert text.startswith(
        "Front left camera: <t=-1.2s>[IMAGE]<t=-0.1s>[IMAGE]<t=0.0s>[IMAGE]Front camera: <t=-2.0s>"
    )
    assert text.count("Front camera: ") == 1
    assert text.count("[IMAGE]") == 22
    assert text.count("<|traj_history|>") == 57
    assert "\nNavigation input is: Continue straight\nYour task: " in text
    assert text.endswith(
        "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    )
    images = [entry for entry in content if entry["type"] == "image"]
    assert [(entry["min_pixels"], entry["max_pixels"]) for entry in images] == [
        (163840, 196608) if camera in (1, 6) else (81920, 98304)
        for camera in data["camera_indices"].tolist()
    ]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("image_frames", torch.zeros(22, 1, 3, 32, 64), "uint8"),
        ("camera_indices", torch.arange(22), "camera-major"),
        ("absolute_timestamps", torch.zeros(22), "int64"),
        ("absolute_timestamps", torch.zeros(21, dtype=torch.int64), "22"),
        ("ego_history_xyz", torch.zeros(1, 16, 3), "20"),
        ("ego_history_rot", torch.zeros(1, 20, 3), "20"),
        ("nav_text", "Continue straight", "one-element"),
        ("nav_text", [""], "nonempty"),
    ],
)
def test_raw_schema_rejects_silent_alignment_errors(field, value, error):
    data = raw_sample()
    data[field] = value
    with pytest.raises((ValueError, TypeError), match=error):
        validate_sample(data)


def test_history_token_count_must_match_poses():
    cfg = config()
    cfg.tokens_per_history_traj = 45
    with pytest.raises(ValueError, match="57"):
        create_messages(raw_sample(), cfg)


def test_time_major_frames_are_sorted_without_losing_pixel_timestamp_association():
    data = raw_sample()
    data["image_frames"][:, 0, 0, 0, 0] = torch.arange(22, dtype=torch.uint8)
    original_ids = data["camera_indices"].clone()
    original_timestamps = data["absolute_timestamps"].clone()
    permutation = torch.tensor(
        [3, 17, 14, 11, 8, 0, 18, 4, 19, 5, 6, 15, 12, 1, 9, 20, 16, 13, 2, 10, 21, 7]
    )
    for field in ("image_frames", "camera_indices", "absolute_timestamps"):
        data[field] = data[field][permutation]
    sorted_data = canonicalize_sample(data)
    assert torch.equal(
        sorted_data["image_frames"][:, 0, 0, 0, 0], torch.arange(22, dtype=torch.uint8)
    )
    assert torch.equal(sorted_data["camera_indices"], original_ids)
    assert torch.equal(sorted_data["absolute_timestamps"], original_timestamps)
    # Preparation should never rewrite the caller's raw capture order.
    assert torch.equal(data["image_frames"][:, 0, 0, 0, 0], permutation.to(torch.uint8))


def test_to_device_keeps_integer_and_boolean_tensors():
    data = {
        "ids": torch.tensor([151655]),
        "mask": torch.tensor([True]),
        "nested": [torch.ones(2), (torch.ones(1),)],
    }
    moved = to_device(data, "cpu", torch.bfloat16)
    assert moved["ids"].dtype == torch.int64
    assert moved["mask"].dtype == torch.bool
    assert moved["nested"][0].dtype == torch.bfloat16
    assert moved["nested"][1][0].dtype == torch.bfloat16


def test_patch_normalization_matches_the_processor_layout():
    from transformers import Cosmos3EdgeImageProcessor

    # Unequal channel statistics catch a wrong channel stride.
    processor = Cosmos3EdgeImageProcessor(image_mean=[0.1, 0.5, 0.9], image_std=[0.2, 0.4, 0.8])
    image = torch.randint(0, 256, (3, 64, 96), dtype=torch.uint8)
    raw = processor(
        images=[image], do_resize=False, do_rescale=False, do_normalize=False, return_tensors="pt"
    )["pixel_values"]
    expected = processor(images=[image], do_resize=False, return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(helper._normalize_patches(raw, processor), expected)


class _TemplateOnlyProcessor:
    image_processor = SimpleNamespace(patch_size=16, merge_size=2)

    def __init__(self, closed: bool):
        self.closed = closed
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        thinking = (
            "<think></think>" if self.closed and not kwargs["enable_thinking"] else "<think>\n"
        )
        return "<|im_start|>assistant\n" + thinking

    def __call__(self, **kwargs):
        self.text = kwargs["text"]
        raise StopIteration


@pytest.mark.parametrize("closed", [True, False])
def test_driving_prompt_disables_thinking(monkeypatch, closed):
    processor = _TemplateOnlyProcessor(closed)
    monkeypatch.setattr(helper, "get_processor", lambda tokenizer, model_config: processor)
    if closed:
        with pytest.raises(StopIteration):
            helper.prepare_model_inputs(raw_sample(), config(), tokenizer=None)
    else:
        with pytest.raises(ValueError, match="<think></think>"):
            helper.prepare_model_inputs(raw_sample(), config(), tokenizer=None)
    assert processor.kwargs["enable_thinking"] is False


def test_vqa_prompt_uses_camera_context_and_question_without_driving_inputs(monkeypatch):
    data = raw_vqa_sample()
    data["absolute_timestamps"][0] -= 60_000  # Retain the actual capture time.
    question = "How should the ego vehicle drive in this scene?"
    messages = build_vqa_conversation(canonicalize_vqa_sample(data), question)
    content = messages[1]["content"]
    text = "".join(entry["text"] for entry in content if entry["type"] == "text")
    images = [entry for entry in content if entry["type"] == "image"]
    assert len(images) == 24
    assert {(entry["min_pixels"], entry["max_pixels"]) for entry in images} == {(163840, 196608)}
    assert text.startswith("Front left camera: <t=-0.4s><t=-0.2s><t=-0.1s><t=0.0s>")
    assert "Front camera: <t=-0.3s>" in text
    assert text.endswith("\n" + question)
    assert "Navigation input" not in text
    assert "Your task:" not in text
    assert "<|traj_" not in text

    processor = _TemplateOnlyProcessor(closed=True)
    monkeypatch.setattr(helper, "get_processor", lambda tokenizer, model_config: processor)
    with pytest.raises(StopIteration):
        helper.prepare_vqa_inputs(data, config(), tokenizer=None, question=question)
    assert processor.text == "<|im_start|>assistant\n"


def test_vqa_sorts_all_24_images_with_their_capture_times():
    data = raw_vqa_sample()
    data["image_frames"][:, 0, 0, 0, 0] = torch.arange(24, dtype=torch.uint8)
    original_timestamps = data["absolute_timestamps"].clone()
    for field in ("image_frames", "camera_indices", "absolute_timestamps"):
        data[field] = data[field].flip(0)
    canonical = canonicalize_vqa_sample(data)
    assert torch.equal(canonical["image_frames"][:, 0, 0, 0, 0], torch.arange(24, dtype=torch.uint8))
    assert torch.equal(canonical["absolute_timestamps"], original_timestamps)
    assert set(canonical["camera_pixel_bounds"].values()) == {(163840, 196608)}
    assert data["image_frames"][0, 0, 0, 0, 0] == 23


def test_vqa_rejects_the_driving_frame_profile():
    with pytest.raises(ValueError, match="24"):
        canonicalize_vqa_sample(raw_sample())


def test_vqa_requires_four_frames_per_camera_and_uniform_pixel_bounds():
    data = raw_vqa_sample()
    data["camera_indices"][0] = 1
    with pytest.raises(ValueError, match="frame counts 4/4/4/4/4/4"):
        validate_vqa_sample(data)
    data = raw_vqa_sample()
    data["camera_pixel_bounds"] = {0: (81920, 98304)}
    with pytest.raises(ValueError, match="163840, 196608"):
        validate_vqa_sample(data)


@pytest.mark.parametrize("question", ["", "  ", None])
def test_vqa_rejects_empty_questions(question):
    with pytest.raises(ValueError, match="question must be a nonempty string"):
        build_vqa_conversation(canonicalize_vqa_sample(raw_vqa_sample()), question)


@pytest.mark.skipif(
    "ALPAMAYO2_EDGE_CHECKPOINT" not in os.environ,
    reason="set ALPAMAYO2_EDGE_CHECKPOINT to a local checkpoint directory",
)
def test_local_checkpoint_prompt_ends_with_closed_thinking():
    from transformers import AutoConfig

    from alpamayo2_edge.config import build_alpamayo2_edge_tokenizer

    path = os.environ["ALPAMAYO2_EDGE_CHECKPOINT"]
    model_config = AutoConfig.from_pretrained(path)
    tokenizer = build_alpamayo2_edge_tokenizer(path)
    inputs = helper.prepare_model_inputs(raw_sample(), model_config, tokenizer)["tokenized_data"]
    text = tokenizer.decode(inputs["input_ids"][0])
    assert text.endswith("<|im_start|>assistant\n<think></think>")
    image_tokens = inputs["input_ids"].eq(model_config.vlm_config.image_token_id)
    torch.testing.assert_close(inputs["mm_token_type_ids"], image_tokens.long())


@pytest.mark.skipif(
    "ALPAMAYO2_EDGE_CHECKPOINT" not in os.environ,
    reason="set ALPAMAYO2_EDGE_CHECKPOINT to a local checkpoint directory",
)
def test_local_checkpoint_vqa_packs_24_images_and_only_removes_assistant_thinking():
    from transformers import AutoConfig

    from alpamayo2_edge.config import build_alpamayo2_edge_tokenizer

    path = os.environ["ALPAMAYO2_EDGE_CHECKPOINT"]
    model_config = AutoConfig.from_pretrained(path)
    tokenizer = build_alpamayo2_edge_tokenizer(path)
    data = raw_vqa_sample()
    # Even if extra driving fields exist, VQA must not insert them into the prompt.
    data["nav_text"] = ["Turn left"]
    data["ego_history_xyz"] = torch.zeros(1, 20, 3)
    question = "What does <think></think> mean in this driving scene?"
    prepared = helper.prepare_vqa_inputs(data, model_config, tokenizer, question)
    assert set(prepared) == {"tokenized_data"}
    vqa = prepared["tokenized_data"]
    text = tokenizer.decode(vqa["input_ids"][0])
    assert text.endswith("<|im_start|>assistant\n")
    assert question in text
    assert text.count("<think></think>") == 1
    assert "<|traj_history" not in text
    assert "Navigation input" not in text
    assert "<|video_pad|>" not in text
    assert text.count("<|vision_start|>") == 24
    assert vqa["image_grid_thw"].shape == (24, 3)
    assert vqa["image_grid_thw"].unique(dim=0).shape[0] == 1
    image_tokens = vqa["input_ids"].eq(model_config.vlm_config.image_token_id)
    torch.testing.assert_close(vqa["mm_token_type_ids"], image_tokens.long())
