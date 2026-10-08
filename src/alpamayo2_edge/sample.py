# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load portable Edge input samples without loading labels or Python objects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

CAMERA_FRAME_COUNTS = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
CAMERA_PIXEL_BOUNDS = {
    camera: (163840, 196608) if camera in (1, 6) else (81920, 98304)
    for camera in CAMERA_FRAME_COUNTS
}
INPUT_TENSOR_KEYS = (
    "image_frames",
    "camera_indices",
    "absolute_timestamps",
    "ego_history_xyz",
    "ego_history_rot",
)


def normalize_camera_pixel_bounds(bounds: Mapping[Any, Any]) -> dict[int, tuple[int, int]]:
    """Restore integer camera keys lost during JSON serialization."""
    if not isinstance(bounds, Mapping):
        raise TypeError("camera_pixel_bounds must be a camera-to-pixel-bounds mapping")
    normalized = {}
    for camera, pair in bounds.items():
        camera = int(camera)
        if camera not in CAMERA_FRAME_COUNTS:
            raise ValueError(f"Unsupported camera ID in camera_pixel_bounds: {camera}")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each camera_pixel_bounds value must be [min_pixels, max_pixels]")
        lower, upper = pair
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, int)
            or not isinstance(upper, int)
            or not 0 < lower <= upper
        ):
            raise ValueError("camera_pixel_bounds must contain positive integer min <= max")
        normalized[camera] = (lower, upper)
    return normalized


def validate_sample(data: Mapping[str, Any], history_num_steps: int = 20) -> None:
    """Check the six-camera, 22-frame Edge inference schema before preprocessing.

    Frames may arrive in any order; preprocessing sorts them by camera and actual
    capture timestamp. Original sample images are 1080x1920, but other positive
    image resolutions are accepted and resized to the same per-camera budgets.
    """
    for key in INPUT_TENSOR_KEYS:
        if not isinstance(data.get(key), torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
    images = data["image_frames"]
    if (
        images.dtype != torch.uint8
        or images.device.type != "cpu"
        or images.ndim != 5
        or images.shape[:3] != (22, 1, 3)
        or min(images.shape[-2:]) <= 0
        or not images.is_contiguous()
    ):
        raise ValueError("image_frames must be contiguous CPU uint8 [22, 1, 3, H, W]")
    cameras = data["camera_indices"]
    if cameras.dtype != torch.int64 or cameras.shape != (22,):
        raise ValueError("camera_indices must be int64 [22]")
    ids, counts = cameras.unique(return_counts=True)
    if dict(zip(ids.tolist(), counts.tolist(), strict=True)) != CAMERA_FRAME_COUNTS:
        raise ValueError(
            "Edge camera-major profile requires IDs 0/1/2/3/5/6 with frame counts 3/5/3/3/3/5"
        )
    timestamps = data["absolute_timestamps"]
    if timestamps.dtype != torch.int64:
        raise TypeError("absolute_timestamps must be int64 microseconds")
    if timestamps.shape not in ((22,), (22, 1)):
        raise ValueError("absolute_timestamps must have shape [22] or [22, 1]")
    for key, tail in (
        ("ego_history_xyz", (history_num_steps, 3)),
        ("ego_history_rot", (history_num_steps, 3, 3)),
    ):
        value = data[key]
        if tuple(value.shape) not in ((1, *tail), (1, 1, *tail)):
            raise ValueError(
                f"{key} must have shape [1, {', '.join(map(str, tail))}] (optionally batched)"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain finite floating-point poses")
    nav = data.get("nav_text")
    if not isinstance(nav, (list, tuple)) or len(nav) != 1 or not isinstance(nav[0], str):
        raise ValueError("nav_text must be a one-element list containing the navigation text")
    if not nav[0].strip():
        raise ValueError("nav_text must contain nonempty navigation text")
    if "camera_pixel_bounds" in data:
        normalize_camera_pixel_bounds(data["camera_pixel_bounds"])


def canonical_frame_order(camera_indices: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Frame indices sorted by camera, then capture timestamp."""
    timestamps = timestamps.reshape(-1)
    # Two stable sorts avoid subtracting large epoch timestamps or assuming a
    # fixed camera cadence. Sorting order is identical to Python (camera, time).
    by_time = torch.argsort(timestamps, stable=True)
    return by_time[torch.argsort(camera_indices[by_time], stable=True)]


def canonicalize_sample(data: Mapping[str, Any], history_num_steps: int = 20) -> dict[str, Any]:
    """Sort all frame-aligned fields together by camera, then capture timestamp."""
    validate_sample(data, history_num_steps=history_num_steps)
    result = dict(data)
    order = canonical_frame_order(data["camera_indices"], data["absolute_timestamps"])
    for key in ("image_frames", "camera_indices", "absolute_timestamps"):
        result[key] = data[key][order]
    result["camera_pixel_bounds"] = {
        **CAMERA_PIXEL_BOUNDS,
        **normalize_camera_pixel_bounds(data.get("camera_pixel_bounds", {})),
    }
    return result


def load_sample(path: str | Path) -> dict[str, Any]:
    """Read ``sample.json`` and raw input safetensors from one sample directory.

    Both ``input.safetensors`` and the checkpoint bundle's ``tensors.safetensors``
    filename are supported. Only the five inference tensor fields are read;
    ``ground_truth.safetensors`` is never opened. The returned dictionary retains
    the capture order from disk. :func:`prepare_model_inputs` performs sorting.
    """
    path = Path(path)
    if path.is_dir():
        folder = path
        tensor_path = folder / "input.safetensors"
        if not tensor_path.is_file():
            tensor_path = folder / "tensors.safetensors"
    else:
        if path.name not in ("input.safetensors", "tensors.safetensors"):
            raise ValueError(
                "Provide a sample directory, input.safetensors, or tensors.safetensors"
            )
        folder, tensor_path = path.parent, path
    with (folder / "sample.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("sample.json must contain a JSON object")
    if set(INPUT_TENSOR_KEYS).intersection(metadata):
        raise ValueError("sample.json must not redefine raw tensor fields")
    data = dict(metadata)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        missing = set(INPUT_TENSOR_KEYS) - set(handle.keys())
        if missing:
            raise ValueError(f"Raw sample is missing required tensors: {sorted(missing)}")
        data.update({key: handle.get_tensor(key) for key in INPUT_TENSOR_KEYS})
    if "camera_pixel_bounds" in data:
        data["camera_pixel_bounds"] = normalize_camera_pixel_bounds(data["camera_pixel_bounds"])
    validate_sample(data)
    return data
