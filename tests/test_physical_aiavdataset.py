"""PhysicalAI-AV boundary tests with real pose mathematics and in-memory media."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from alpamayo2_edge.load_physical_aiavdataset import (
    load_physical_aiavdataset,
    load_physical_aiavdataset_vqa,
)
from alpamayo2_edge.sample import validate_sample, validate_vqa_sample

CAMERAS = {
    "CAMERA_CROSS_LEFT_120FOV": 0,
    "CAMERA_FRONT_WIDE_120FOV": 1,
    "CAMERA_CROSS_RIGHT_120FOV": 2,
    "CAMERA_REAR_LEFT_70FOV": 3,
    "CAMERA_REAR_RIGHT_70FOV": 5,
    "CAMERA_FRONT_TELE_30FOV": 6,
}


class Camera:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.timestamps = np.array([0, 10_000_000], dtype=np.int64)
        self.requests = []
        self.mutate = lambda frames, times: (frames, times)

    def decode_images_from_timestamps(self, timestamps):
        self.requests.append(timestamps.copy())
        count = len(timestamps)
        frames = np.zeros((count, 4, 8, 3), dtype=np.uint8)
        frames[:, :, :, 0] = (self.camera_id * 10 + np.arange(count))[:, None, None]
        # The SDK returns the closest frame at or before each requested time.
        actual = timestamps + np.array([-17, -9, -23, -13, -19], dtype=np.int64)[:count]
        actual = np.maximum(actual, 0)
        return self.mutate(frames, actual)


class Egomotion:
    def __init__(self):
        self.timestamps = np.array([0, 10_000_000], dtype=np.int64)
        self.requests = []

    @property
    def time_range(self):
        return self.timestamps[0], self.timestamps[-1]

    def __call__(self, timestamps):
        self.requests.append(timestamps.copy())
        relative_seconds = (timestamps - 5_100_000) / 1_000_000
        local_positions = relative_seconds[:, None] * np.array([1.0, -2.0, 0.5])
        world_from_t0 = Rotation.from_euler("xyz", [20.0, -10.0, 90.0], degrees=True)
        rotation_in_t0 = Rotation.from_euler(
            "x", relative_seconds[:, None] * (-30 / 1.9), degrees=True
        )
        return SimpleNamespace(
            pose=SimpleNamespace(
                translation=np.array([100.0, -80.0, 7.0]) + world_from_t0.apply(local_positions),
                rotation=world_from_t0 * rotation_in_t0,
            )
        )


class Dataset:
    def __init__(self):
        self.features = SimpleNamespace(
            CAMERA=SimpleNamespace(**{name: "camera/" + name.lower() for name in CAMERAS}),
            LABELS=SimpleNamespace(EGOMOTION="labels/egomotion"),
        )
        self.cameras = {camera_id: Camera(camera_id) for camera_id in CAMERAS.values()}
        self.ego = Egomotion()
        self.requests = []

    def get_clip_feature(self, clip_id, feature, *, maybe_stream):
        self.requests.append((clip_id, feature, maybe_stream))
        if feature == "labels/egomotion":
            return self.ego
        camera_id = CAMERAS[feature.split("/")[-1].upper()]
        return self.cameras[camera_id]


def test_edge_camera_schedule_preserves_decoded_timestamps_and_image_association():
    dataset = Dataset()
    sample = load_physical_aiavdataset(
        "clip-123", nav_text="Turn left", avdi=dataset, maybe_stream=False
    )
    validate_sample(sample)
    assert sample["image_frames"].shape == (22, 1, 3, 4, 8)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["image_frames"].is_contiguous()
    assert sample["camera_indices"].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
    ]
    assert sample["image_frames"][:, 0, 0, 0, 0].tolist() == [
        0,
        1,
        2,
        10,
        11,
        12,
        13,
        14,
        20,
        21,
        22,
        30,
        31,
        32,
        50,
        51,
        52,
        60,
        61,
        62,
        63,
        64,
    ]
    assert dataset.cameras[0].requests[0].tolist() == [4_100_000, 5_000_000, 5_100_000]
    assert dataset.cameras[1].requests[0].tolist() == [
        3_100_000,
        4_100_000,
        4_600_000,
        5_000_000,
        5_100_000,
    ]
    assert dataset.cameras[6].requests[0].tolist() == [
        3_100_000,
        4_100_000,
        4_600_000,
        5_000_000,
        5_100_000,
    ]
    assert sample["absolute_timestamps"].dtype == torch.int64
    assert sample["absolute_timestamps"][:8].tolist() == [
        4_099_983,
        4_999_991,
        5_099_977,
        3_099_983,
        4_099_991,
        4_599_977,
        4_999_987,
        5_099_981,
    ]
    assert sample["nav_text"] == ["Turn left"]
    assert sample["clip_id"] == "clip-123"
    assert sample["t0_us"] == 5_100_000
    assert sample["camera_pixel_bounds"][0] == (81920, 98304)
    assert sample["camera_pixel_bounds"][1] == (163840, 196608)
    assert all(request[0] == "clip-123" and request[2] is False for request in dataset.requests)


def test_history_uses_full_rigid_transform_at_t0_without_querying_future():
    dataset = Dataset()
    sample = load_physical_aiavdataset("clip-123", nav_text="Turn left", avdi=dataset)
    assert len(dataset.ego.requests) == 1
    assert dataset.ego.requests[0].tolist() == [
        3_200_000,
        3_300_000,
        3_400_000,
        3_500_000,
        3_600_000,
        3_700_000,
        3_800_000,
        3_900_000,
        4_000_000,
        4_100_000,
        4_200_000,
        4_300_000,
        4_400_000,
        4_500_000,
        4_600_000,
        4_700_000,
        4_800_000,
        4_900_000,
        5_000_000,
        5_100_000,
    ]
    assert sample["ego_history_xyz"].shape == (1, 20, 3)
    assert sample["ego_history_rot"].shape == (1, 20, 3, 3)
    torch.testing.assert_close(sample["ego_history_xyz"][0, 0], torch.tensor([-1.9, 3.8, -0.95]))
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        sample["ego_history_rot"][0, 0],
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, np.sqrt(3) / 2, -0.5], [0.0, 0.5, np.sqrt(3) / 2]],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(sample["ego_history_rot"][0, -1], torch.eye(3), atol=1e-6, rtol=0)
    assert not any("future" in key for key in sample)


def test_vqa_uses_six_cameras_with_four_frames_and_actual_capture_times():
    dataset = Dataset()
    dataset.revision = "test-dataset-revision"
    del dataset.features.LABELS
    sample = load_physical_aiavdataset_vqa("clip-123", avdi=dataset, maybe_stream=False)

    validate_vqa_sample(sample)
    assert sample["image_frames"].shape == (24, 1, 3, 4, 8)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["image_frames"].is_contiguous()
    assert sample["camera_indices"].tolist() == [
        camera_id for camera_id in CAMERAS.values() for _ in range(4)
    ]
    assert sample["image_frames"][:, 0, 0, 0, 0].tolist() == [
        camera_id * 10 + frame_index for camera_id in CAMERAS.values() for frame_index in range(4)
    ]
    for camera in dataset.cameras.values():
        assert len(camera.requests) == 1
        assert camera.requests[0].tolist() == [4_800_000, 4_900_000, 5_000_000, 5_100_000]
    assert sample["absolute_timestamps"].dtype == torch.int64
    assert (
        sample["absolute_timestamps"].tolist() == [4_799_983, 4_899_991, 4_999_977, 5_099_987] * 6
    )
    assert sample["camera_pixel_bounds"] == {
        camera_id: (163840, 196608) for camera_id in CAMERAS.values()
    }
    assert sample["min_pixels"] == 163840
    assert sample["max_pixels"] == 196608
    assert sample["input_profile"] == "vqa_6cam_4f"
    assert sample["clip_id"] == "clip-123"
    assert sample["t0_us"] == 5_100_000
    assert sample["provenance"] == {
        "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "clip_id": "clip-123",
        "t0_us": 5_100_000,
        "dataset_revision": "test-dataset-revision",
    }
    assert not {"nav_text", "ego_history_xyz", "ego_history_rot"} & sample.keys()
    assert dataset.ego.requests == []
    assert len(dataset.requests) == 6
    assert all(request[0] == "clip-123" and request[2] is False for request in dataset.requests)


@pytest.mark.parametrize("t0_us", [True, 5_100_000.0, -1, 299_999, 2**63])
def test_vqa_invalid_anchor_is_rejected_before_dataset_access(t0_us):
    dataset = Dataset()
    with pytest.raises((TypeError, ValueError), match="t0_us"):
        load_physical_aiavdataset_vqa("clip-123", t0_us, avdi=dataset)
    assert dataset.requests == []


@pytest.mark.parametrize("clip_id", ["", "  ", None])
def test_vqa_invalid_clip_is_rejected_before_dataset_access(clip_id):
    dataset = Dataset()
    with pytest.raises(ValueError, match="clip_id"):
        load_physical_aiavdataset_vqa(clip_id, avdi=dataset)
    assert dataset.requests == []


def test_vqa_anchor_accepts_inclusive_clip_start_and_numpy_integer():
    sample = load_physical_aiavdataset_vqa("clip-123", np.int64(300_000), avdi=Dataset())
    assert sample["t0_us"] == 300_000
    assert sample["absolute_timestamps"][::4].tolist() == [0] * 6


@pytest.mark.parametrize("bounds", [[4_800_001, 10_000_000], [0, 5_099_999]])
def test_vqa_requests_must_fit_camera_time_bounds(bounds):
    dataset = Dataset()
    dataset.cameras[0].timestamps = np.array(bounds, dtype=np.int64)
    with pytest.raises(ValueError, match="CAMERA_CROSS_LEFT_120FOV.*range"):
        load_physical_aiavdataset_vqa("clip-123", avdi=dataset)
    assert dataset.cameras[0].requests == []


@pytest.mark.parametrize("nav_text", ["", "  ", None, ["Go straight"]])
def test_missing_navigation_is_rejected_before_dataset_access(nav_text):
    dataset = Dataset()
    with pytest.raises(ValueError, match="nav_text"):
        load_physical_aiavdataset("clip-123", nav_text=nav_text, avdi=dataset)
    assert dataset.requests == []


@pytest.mark.parametrize("t0_us", [True, 5_100_000.0, -1, 1_999_999, 2**63])
def test_invalid_anchor_time_is_rejected_before_dataset_access(t0_us):
    dataset = Dataset()
    with pytest.raises((TypeError, ValueError), match="t0_us"):
        load_physical_aiavdataset("clip-123", t0_us, nav_text="Straight", avdi=dataset)
    assert dataset.requests == []


def test_two_second_anchor_accepts_inclusive_clip_start():
    sample = load_physical_aiavdataset("clip-123", 2_000_000, nav_text="Straight", avdi=Dataset())
    assert sample["absolute_timestamps"][3].item() == 0


@pytest.mark.parametrize("feature", ["ego", "camera"])
def test_requested_history_must_be_within_feature_time_bounds(feature):
    dataset = Dataset()
    if feature == "ego":
        dataset.ego.timestamps = np.array([3_200_001, 10_000_000], dtype=np.int64)
    else:
        dataset.cameras[1].timestamps = np.array([3_100_001, 10_000_000], dtype=np.int64)
    with pytest.raises(ValueError, match="range|bounds"):
        load_physical_aiavdataset("clip-123", nav_text="Straight", avdi=dataset)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frames, times: (frames.astype(np.float32), times), "uint8"),
        (lambda frames, times: (frames[:-1], times), "frames"),
        (lambda frames, times: (frames[..., :2], times), "frames"),
        (lambda frames, times: (frames, times.astype(np.float64)), "int64"),
        (lambda frames, times: (frames, times[:-1]), "timestamps"),
        (lambda frames, times: (frames, times + 1_000_000), "timestamps"),
    ],
)
@pytest.mark.parametrize("vqa", [False, True])
def test_malformed_decoder_outputs_fail_with_context(mutation, match, vqa):
    dataset = Dataset()
    dataset.cameras[0].mutate = mutation
    with pytest.raises((ValueError, TypeError), match=match):
        if vqa:
            load_physical_aiavdataset_vqa("clip-123", avdi=dataset)
        else:
            load_physical_aiavdataset("clip-123", nav_text="Straight", avdi=dataset)


@pytest.mark.parametrize("vqa", [False, True])
def test_missing_optional_sdk_has_installation_hint(monkeypatch, vqa):
    original_import = builtins.__import__

    def without_dataset(name, *args, **kwargs):
        if name == "physical_ai_av":
            raise ModuleNotFoundError("No module named 'physical_ai_av'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_dataset)
    with pytest.raises(ImportError, match=r"alpamayo2-edge\[dataset\]"):
        if vqa:
            load_physical_aiavdataset_vqa("clip-123")
        else:
            load_physical_aiavdataset("clip-123", nav_text="Straight")
