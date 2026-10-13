# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the trained Edge input profile from the optional PhysicalAI-AV SDK."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import torch

from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS, validate_sample

_CAMERAS = (
    (0, "CAMERA_CROSS_LEFT_120FOV"),
    (1, "CAMERA_FRONT_WIDE_120FOV"),
    (2, "CAMERA_CROSS_RIGHT_120FOV"),
    (3, "CAMERA_REAR_LEFT_70FOV"),
    (5, "CAMERA_REAR_RIGHT_70FOV"),
    (6, "CAMERA_FRONT_TELE_30FOV"),
)
_FRONT_OFFSETS_US = np.array([-2_000_000, -1_000_000, -500_000, -100_000, 0], dtype=np.int64)
_OTHER_OFFSETS_US = np.array([-1_000_000, -100_000, 0], dtype=np.int64)


def _check_time_bounds(requested: np.ndarray, bounds: tuple, feature: str) -> None:
    lower, upper = bounds
    if requested[0] < lower or requested[-1] > upper:
        raise ValueError(
            f"{feature} requested timestamps [{requested[0]}, {requested[-1]}] are outside "
            f"the available inclusive range [{lower}, {upper}] microseconds"
        )


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    *,
    nav_text: str,
    avdi: Any = None,
    maybe_stream: bool = True,
) -> dict[str, Any]:
    """Read six cameras and 20 past ego poses as one raw Edge inference sample.

    ``t0_us`` and all camera timestamps use clip-relative integer microseconds.
    Front cameras 1/6 sample at -2/-1/-0.5/-0.1/0 seconds; the other four sample
    at -1/-0.1/0 seconds. Actual decoder timestamps are retained. ``ego_t0_us``
    is their maximum across all 22 frames; history poses span -1.9 through 0
    seconds relative to it and use the ego frame at that actual capture time.
    The returned ``t0_us`` remains the requested camera-sampling anchor.

    ``nav_text`` must be supplied explicitly: the public dataset does not supply
    the route instruction. No future trajectory or calibration is requested.
    Pass an existing SDK interface through ``avdi`` to reuse its cache/revision.
    """
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("clip_id must be a nonempty string")
    if not isinstance(nav_text, str) or not nav_text.strip():
        raise ValueError("nav_text must be an explicit nonempty navigation instruction")
    if isinstance(t0_us, bool) or not isinstance(t0_us, Integral):
        raise TypeError("t0_us must be an integer number of clip-relative microseconds")
    if not 2_000_000 <= t0_us <= np.iinfo(np.int64).max:
        raise ValueError("t0_us must fit int64 and allow the full two-second camera history")
    t0_us = int(t0_us)
    if avdi is None:
        try:
            import physical_ai_av
        except ModuleNotFoundError as exc:
            if exc.name != "physical_ai_av":
                raise
            raise ImportError(
                "PhysicalAI-AV loading requires the optional dataset extra; "
                "install it with pip install 'alpamayo2-edge[dataset]'."
            ) from exc
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    image_groups, camera_groups, timestamp_groups = [], [], []
    image_size = None
    for camera_id, camera_name in _CAMERAS:
        feature = getattr(avdi.features.CAMERA, camera_name)
        camera = avdi.get_clip_feature(clip_id, feature, maybe_stream=maybe_stream)
        offsets = _FRONT_OFFSETS_US if camera_id in (1, 6) else _OTHER_OFFSETS_US
        requested = t0_us + offsets
        available = np.asarray(camera.timestamps)
        if (
            available.ndim != 1
            or available.size == 0
            or available.dtype != np.int64
            or np.any(available[1:] < available[:-1])
        ):
            raise ValueError(f"{camera_name} available timestamps must be sorted nonempty int64")
        _check_time_bounds(requested, (available[0], available[-1]), camera_name)
        frames, timestamps = camera.decode_images_from_timestamps(requested)
        frames, timestamps = np.asarray(frames), np.asarray(timestamps)
        count = len(requested)
        if (
            frames.ndim != 4
            or frames.shape[0] != count
            or frames.shape[-1] != 3
            or min(frames.shape[1:3]) <= 0
        ):
            raise ValueError(f"{camera_name} decoded frames must have shape [{count}, H, W, 3]")
        if frames.dtype != np.uint8:
            raise TypeError(f"{camera_name} decoded frames must be uint8 RGB")
        if timestamps.shape != (count,) or timestamps.dtype != np.int64:
            raise ValueError(f"{camera_name} decoded timestamps must be int64 [{count}]")
        if (
            np.any(timestamps < available[0])
            or np.any(timestamps > requested)
            or np.any(timestamps[1:] < timestamps[:-1])
        ):
            raise ValueError(
                f"{camera_name} decoded timestamps must be ordered, in range, and at or before each request"
            )
        if image_size is not None and image_size != frames.shape[1:3]:
            raise ValueError("Decoded camera frames must have the same height and width")
        image_size = frames.shape[1:3]
        image_groups.append(torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(1))
        camera_groups.append(torch.full((count,), camera_id, dtype=torch.int64))
        timestamp_groups.append(torch.from_numpy(timestamps))

    absolute_timestamps = torch.cat(timestamp_groups)
    ego_t0_us = int(absolute_timestamps.max().item())
    egomotion = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=maybe_stream
    )
    history_timestamps = ego_t0_us + np.arange(-1_900_000, 1, 100_000, dtype=np.int64)
    _check_time_bounds(history_timestamps, egomotion.time_range, "egomotion")
    history = egomotion(history_timestamps)
    world_xyz = np.asarray(history.pose.translation)
    world_rot = history.pose.rotation
    matrices = world_rot.as_matrix()
    if (
        world_xyz.shape != (20, 3)
        or matrices.shape != (20, 3, 3)
        or not np.isfinite(world_xyz).all()
        or not np.isfinite(matrices).all()
    ):
        raise ValueError("egomotion must provide 20 finite XYZ positions and 3x3 rotations")
    ego_t0_inverse = world_rot[-1].inv()
    local_xyz = ego_t0_inverse.apply(world_xyz - world_xyz[-1])
    local_rot = (ego_t0_inverse * world_rot).as_matrix()

    provenance = {
        "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "clip_id": clip_id,
        "t0_us": t0_us,
        "ego_t0_us": ego_t0_us,
        "ego_t0_source": "max_captured_image_timestamp",
    }
    revision = getattr(avdi, "revision", None)
    if revision is not None:
        provenance["dataset_revision"] = str(revision)
    data = {
        "image_frames": torch.cat(image_groups).contiguous(),
        "camera_indices": torch.cat(camera_groups),
        "absolute_timestamps": absolute_timestamps,
        "ego_history_xyz": torch.from_numpy(local_xyz).float().unsqueeze(0),
        "ego_history_rot": torch.from_numpy(local_rot).float().unsqueeze(0),
        "nav_text": [nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "min_pixels": 163840,
        "max_pixels": 196608,
        "input_profile": "camera_subsampled",
        "history_num_steps": 20,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "ego_t0_us": ego_t0_us,
        "provenance": provenance,
    }
    validate_sample(data)
    return data
