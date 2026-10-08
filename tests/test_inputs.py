"""Input contracts that protect camera, timestamp, and trajectory alignment."""

from types import SimpleNamespace

import pytest
import torch

from alpamayo2_edge.helper import create_messages, to_device
from alpamayo2_edge.sample import canonicalize_sample, validate_sample


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
