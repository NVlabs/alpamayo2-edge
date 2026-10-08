"""Regressions for capture-time precision and its connection to image tokens."""

import re

import torch
from test_inputs import config, raw_sample

from alpamayo2_edge.chat_template import relative_time_labels
from alpamayo2_edge.helper import create_messages
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


def test_capture_time_changes_reach_prompt_and_images_follow_reordered_times():
    raw = raw_sample()
    raw["image_frames"][:, 0, 0, 0, 0] = torch.arange(22, dtype=torch.uint8)

    def prompt_labels_and_images(data):
        content = create_messages(data, config())[1]["content"]
        text = "".join(entry["text"] for entry in content if entry["type"] == "text")
        images = torch.stack([entry["image"] for entry in content if entry["type"] == "image"])
        return re.findall(r"<t=[^>]+>", text), images

    baseline_labels, baseline_images = prompt_labels_and_images(raw)
    changed = dict(raw, absolute_timestamps=raw["absolute_timestamps"].clone())
    changed["absolute_timestamps"][0] -= 160_000
    perturbed_labels, perturbed_images = prompt_labels_and_images(changed)
    assert baseline_labels[0] != perturbed_labels[0]
    assert baseline_labels[1:] == perturbed_labels[1:]
    assert torch.equal(baseline_images, perturbed_images)

    # Move that same image after the other two captures of camera 0. The prompt
    # must keep each image aligned with its corresponding timestamp label.
    changed["absolute_timestamps"][0] = raw["absolute_timestamps"][2] + 200_000
    reordered_labels, reordered_images = prompt_labels_and_images(changed)
    assert reordered_labels[:2] == baseline_labels[1:3]
    assert reordered_labels[2] != baseline_labels[0]
    assert reordered_labels[3:] == baseline_labels[3:]
    expected_order = [1, 2, 0, *range(3, 22)]
    assert torch.equal(reordered_images, baseline_images[expected_order])
