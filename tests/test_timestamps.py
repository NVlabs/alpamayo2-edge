"""Regressions for capture-time precision and its connection to image tokens."""

import json
import os
import re
from pathlib import Path

import pytest
import torch
from test_inputs import raw_sample

from alpamayo2_edge.chat_template import relative_time_labels
from alpamayo2_edge.sample import canonicalize_sample


def test_epoch_microseconds_keep_subsecond_precision_and_ignore_time_origin():
    offsets = torch.tensor([-1_160_000, -110_000, -11_000, 0], dtype=torch.int64)
    expected = ["<t=-1.2s>", "<t=-0.1s>", "<t=0.0s>", "<t=0.0s>"]
    assert relative_time_labels(offsets) == expected
    assert relative_time_labels(offsets + 1_700_000_000_000_000) == expected


def test_timestamp_column_shape_and_negative_zero_match_vector_labels():
    timestamps = torch.tensor(
        [1_700_000_000_000_000 - 11_000, 1_700_000_000_000_000], dtype=torch.int64
    )
    assert relative_time_labels(timestamps) == ["<t=0.0s>", "<t=0.0s>"]
    assert relative_time_labels(timestamps[:, None]) == relative_time_labels(timestamps)
    data = raw_sample()
    vector = canonicalize_sample(data)
    data["absolute_timestamps"] = data["absolute_timestamps"][:, None]
    column = canonicalize_sample(data)
    assert column["absolute_timestamps"].shape == (22, 1)
    assert relative_time_labels(column["absolute_timestamps"]) == relative_time_labels(
        vector["absolute_timestamps"]
    )


def test_reference_is_latest_last_camera_capture_even_if_another_camera_is_later():
    data = raw_sample()
    # The newest front-left capture is later than the newest telephoto capture.
    data["absolute_timestamps"][2] += 200_000
    data["image_frames"][:, 0, 0, 0, 0] = torch.arange(22, dtype=torch.uint8)
    capture_order = torch.argsort(data["absolute_timestamps"], stable=True)
    for key in ("image_frames", "camera_indices", "absolute_timestamps"):
        data[key] = data[key][capture_order]
    canonical = canonicalize_sample(data)
    labels = relative_time_labels(canonical["absolute_timestamps"])
    assert canonical["camera_indices"][-1].item() == 6
    assert labels[2] == "<t=0.2s>"
    assert labels[-1] == "<t=0.0s>"
    assert canonical["image_frames"][2, 0, 0, 0, 0].item() == 2
    assert canonical["image_frames"][-1, 0, 0, 0, 0].item() == 21


@pytest.mark.skipif(
    not os.environ.get("ALPAMAYO2_EDGE_TEST_BUNDLE"),
    reason="Set ALPAMAYO2_EDGE_TEST_BUNDLE to exercise actual checkpoint tokenization",
)
def test_capture_time_changes_reach_input_ids_and_images_follow_reordered_times():
    from transformers import AutoTokenizer

    from alpamayo2_edge.config import Alpamayo2EdgeConfig, normalize_checkpoint_config
    from alpamayo2_edge.helper import prepare_model_inputs
    from alpamayo2_edge.sample import load_sample

    bundle = Path(os.environ["ALPAMAYO2_EDGE_TEST_BUNDLE"]).resolve()
    cfg = Alpamayo2EdgeConfig(
        **normalize_checkpoint_config(json.loads((bundle / "model/config.json").read_text()))
    )
    cfg._name_or_path = str(bundle / "model")
    tokenizer = AutoTokenizer.from_pretrained(bundle / "model")
    raw = load_sample(bundle / "samples/sample_000")
    canonical = canonicalize_sample(raw)
    baseline = prepare_model_inputs(canonical, cfg, tokenizer)["tokenized_data"]

    def decoded_labels(packed):
        text = tokenizer.decode(packed["input_ids"][0], skip_special_tokens=False)
        return re.findall(r"<t=[^>]+>", text)

    baseline_labels = decoded_labels(baseline)
    changed = dict(canonical, absolute_timestamps=canonical["absolute_timestamps"].clone())
    changed["absolute_timestamps"][0] -= 160_000
    perturbed = prepare_model_inputs(changed, cfg, tokenizer)["tokenized_data"]
    perturbed_labels = decoded_labels(perturbed)
    assert baseline_labels[0] != perturbed_labels[0]
    assert baseline_labels[1:] == perturbed_labels[1:]
    assert not torch.equal(baseline["input_ids"], perturbed["input_ids"])
    assert torch.equal(baseline["pixel_values"], perturbed["pixel_values"])
    assert torch.equal(baseline["image_grid_thw"], perturbed["image_grid_thw"])
    assert torch.equal(baseline["mm_token_type_ids"], perturbed["mm_token_type_ids"])

    # Move that same image after the other two captures of camera 0. The actual
    # processor's image patches must follow the corresponding timestamp label.
    changed["absolute_timestamps"][0] = canonical["absolute_timestamps"][2] + 200_000
    reordered = prepare_model_inputs(changed, cfg, tokenizer)["tokenized_data"]
    reordered_labels = decoded_labels(reordered)
    assert reordered_labels[:2] == baseline_labels[1:3]
    assert reordered_labels[2] != baseline_labels[0]
    assert reordered_labels[3:] == baseline_labels[3:]
    baseline_chunks = baseline["pixel_values"].split(
        baseline["image_grid_thw"].prod(dim=1).tolist()
    )
    reordered_chunks = reordered["pixel_values"].split(
        reordered["image_grid_thw"].prod(dim=1).tolist()
    )
    expected_order = [1, 2, 0, *range(3, 22)]
    for actual, source_index in zip(reordered_chunks, expected_order, strict=True):
        assert torch.equal(actual, baseline_chunks[source_index])
