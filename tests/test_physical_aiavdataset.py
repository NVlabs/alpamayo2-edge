"""PhysicalAI-AV boundary tests with real pose mathematics and in-memory media."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_edge.sample import validate_sample

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
        self.capture_lag_us = None

    def decode_images_from_timestamps(self, timestamps):
        self.requests.append(timestamps.copy())
        count = len(timestamps)
        frames = np.zeros((count, 4, 8, 3), dtype=np.uint8)
        frames[:, :, :, 0] = (self.camera_id * 10 + np.arange(count))[:, None, None]
        # The SDK returns the closest frame at or before each requested time.
        if self.capture_lag_us is None:
            actual = timestamps + np.array([-17, -9, -23, -13, -19], dtype=np.int64)[:count]
        else:
            actual = timestamps - self.capture_lag_us
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
    assert sample["ego_t0_us"] == 5_099_981
    assert sample["provenance"]["t0_us"] == 5_100_000
    assert sample["provenance"]["ego_t0_us"] == 5_099_981
    assert sample["provenance"]["ego_t0_source"] == "max_captured_image_timestamp"
    assert sample["camera_pixel_bounds"][0] == (81920, 98304)
    assert sample["camera_pixel_bounds"][1] == (163840, 196608)
    assert all(request[0] == "clip-123" and request[2] is False for request in dataset.requests)


def test_history_uses_full_rigid_transform_when_actual_and_requested_anchors_match():
    dataset = Dataset()
    for camera in dataset.cameras.values():
        camera.capture_lag_us = 0
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


@pytest.mark.parametrize("newest_camera", [0, 2, 6])
def test_history_uses_latest_capture_across_all_cameras_and_its_full_rigid_frame(newest_camera):
    dataset = Dataset()
    for camera in dataset.cameras.values():
        camera.capture_lag_us = 50_000
    dataset.cameras[newest_camera].capture_lag_us = 31_478
    sample = load_physical_aiavdataset("clip-123", nav_text="Turn left", avdi=dataset)

    assert sample["t0_us"] == 5_100_000
    assert sample["ego_t0_us"] == 5_068_522
    assert type(sample["ego_t0_us"]) is int
    assert type(sample["provenance"]["ego_t0_us"]) is int
    assert sample["absolute_timestamps"].max().item() == 5_068_522
    # Camera acquisition still uses the requested time, including the front -2s frame.
    assert all(camera.requests[0][-1] == 5_100_000 for camera in dataset.cameras.values())
    assert dataset.cameras[1].requests[0][0] == 3_100_000
    assert len(dataset.ego.requests) == 1
    history_times = dataset.ego.requests[0]
    assert history_times.shape == (20,)
    assert history_times[0] == 3_168_522
    assert history_times[-1] == 5_068_522
    np.testing.assert_array_equal(np.diff(history_times), np.full(19, 100_000))

    # At the actual anchor the fixture has an extra +0.497021-degree roll.
    # Applying its inverse to [-1.9, 3.8, -0.95] gives this hand-derived position.
    torch.testing.assert_close(
        sample["ego_history_xyz"][0, 0],
        torch.tensor([-1.9, 3.7916162088454834, -0.9829275277558354]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(sample["ego_history_rot"][0, -1], torch.eye(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        sample["ego_history_rot"][0, 0],
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, np.sqrt(3) / 2, -0.5], [0.0, 0.5, np.sqrt(3) / 2]],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize(
    "requested_t0,ego_t0,history_start",
    [(5_100_000, 5_068_522, 3_168_522), (2_000_000, 1_968_522, 68_522)],
)
def test_actual_history_accepts_inclusive_bounds_even_when_nominal_anchor_is_outside(
    requested_t0, ego_t0, history_start
):
    dataset = Dataset()
    for camera in dataset.cameras.values():
        camera.capture_lag_us = 31_478
    dataset.ego.timestamps = np.array([history_start, ego_t0], dtype=np.int64)
    sample = load_physical_aiavdataset("clip-123", requested_t0, nav_text="Straight", avdi=dataset)
    assert sample["t0_us"] == requested_t0
    assert sample["ego_t0_us"] == ego_t0
    assert dataset.ego.requests[0][0] == history_start
    assert dataset.ego.requests[0][-1] == ego_t0


def test_actual_history_outside_lower_bound_is_rejected_without_interpolation():
    dataset = Dataset()
    for camera in dataset.cameras.values():
        camera.capture_lag_us = 31_478
    # The old nominal window [3.2s, 5.1s] fits; the actual window starts earlier.
    dataset.ego.timestamps = np.array([3_168_523, 10_000_000], dtype=np.int64)
    with pytest.raises(ValueError, match="egomotion.*3168522.*range"):
        load_physical_aiavdataset("clip-123", nav_text="Straight", avdi=dataset)
    assert dataset.ego.requests == []


def test_camera_request_before_first_capture_still_fails_without_fallback():
    dataset = Dataset()
    dataset.cameras[6].timestamps = np.array([11, 10_000_000], dtype=np.int64)
    with pytest.raises(ValueError, match="CAMERA_FRONT_TELE_30FOV.*range"):
        load_physical_aiavdataset("clip-123", 2_000_000, nav_text="Straight", avdi=dataset)
    assert dataset.cameras[6].requests == []


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
def test_malformed_decoder_outputs_fail_with_context(mutation, match):
    dataset = Dataset()
    dataset.cameras[0].mutate = mutation
    with pytest.raises((ValueError, TypeError), match=match):
        load_physical_aiavdataset("clip-123", nav_text="Straight", avdi=dataset)


def test_missing_optional_sdk_has_installation_hint(monkeypatch):
    original_import = builtins.__import__

    def without_dataset(name, *args, **kwargs):
        if name == "physical_ai_av":
            raise ModuleNotFoundError("No module named 'physical_ai_av'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_dataset)
    with pytest.raises(ImportError, match=r"alpamayo2-edge\[dataset\]"):
        load_physical_aiavdataset("clip-123", nav_text="Straight")
