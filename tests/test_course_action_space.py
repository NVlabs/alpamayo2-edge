"""Check course-heading action space outputs against a fixed reference.

``data/course_action_space_reference.safetensors`` stores frozen float64 encode and decode
outputs for the synthetic scenarios below.
"""

import math
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).parent / "data" / "course_action_space_reference.safetensors"
HISTORY_STEPS = 20
FUTURE_STEPS = 64
DT = 0.1

# Edge action-space configs: the course space and the yaw space.
_COMMON = {
    "accel_mean": 0.02902694707164455,
    "accel_std": 0.6810426736454882,
    "curvature_mean": 0.0002692167976330542,
    "curvature_std": 0.026148280660833106,
    "accel_bounds": (-9.8, 9.8),
    "curvature_bounds": (-0.33, 0.33),
    "dt": DT,
    "n_waypoints": FUTURE_STEPS,
    "theta_ridge": 1e-8,
    "v_lambda": 1e-6,
    "v_ridge": 1e-4,
    "a_lambda": 1e-4,
    "a_ridge": 1e-4,
    "kappa_lambda": 1e-4,
    "kappa_ridge": 1e-4,
}
YAW_CFG = {**_COMMON, "theta_lambda": 1e-6}
COURSE_CFG = {**_COMMON, "theta_lambda": 3e-6, "slip_gradient": 2.5e-3}

# Float64 solves agree across LAPACK builds to ~1e-12. Float32 solves of these smoothing systems
# differ from float64 by up to ~1e-3 normalized action units and vary with the LAPACK kernel, so
# float32 (the inference precision) is checked against the float64 reference at these bounds.
TOLERANCES = {
    torch.float64: {"action": 1e-9, "xyz": 1e-9, "yaw": 1e-9, "t0": 1e-9},
    torch.float32: {"action": 5e-3, "xyz": 5e-4, "yaw": 1e-4, "t0": 5e-5},
}


def _hash_noise(index: torch.Tensor, seed: float) -> torch.Tensor:
    """Deterministic noise in [-1, 1] that does not depend on a torch RNG implementation."""
    return 2.0 * torch.frac(torch.sin(index * 12.9898 + seed) * 43758.5453).abs() - 1.0


# Each scenario: speed(t) [m/s], curvature(t) [1/m] along the course, and optional pose defects.
SCENARIOS = {
    "straight_cruise": dict(speed=lambda t: 15.0 + 0 * t, curvature=lambda t: 0 * t),
    "left_arc": dict(speed=lambda t: 10.0 + 0 * t, curvature=lambda t: 0.02 + 0 * t),
    "right_curve_accelerating": dict(
        speed=lambda t: 4.0 + 1.2 * t, curvature=lambda t: -0.04 + 0 * t
    ),
    "pull_away_full_lock": dict(speed=lambda t: t.clamp(min=0.0), curvature=lambda t: 0.2 + 0 * t),
    "stop_and_go": dict(
        speed=lambda t: torch.where(
            t < 3.0, (2.0 - t).clamp(min=0.0), (1.5 * (t - 4.0)).clamp(min=0.0)
        ),
        curvature=lambda t: 0.03 + 0 * t,
    ),
    "creep_course_fade": dict(speed=lambda t: 0.75 + 0 * t, curvature=lambda t: 0.1 + 0 * t),
    "reversing": dict(speed=lambda t: -2.0 + 0 * t, curvature=lambda t: 0.05 + 0 * t),
    "u_turn_wrap": dict(
        speed=lambda t: 5.0 + 0 * t, curvature=lambda t: torch.where(t > 0, 0.15, 0.0)
    ),
    "lane_change": dict(
        speed=lambda t: 20.0 + 0 * t, curvature=lambda t: 0.004 * torch.sin(2 * math.pi * t / 4)
    ),
    "history_pose_glitch": dict(
        speed=lambda t: 8.0 + 0 * t, curvature=lambda t: 0 * t, history_shift=0.1
    ),
    "future_pose_glitch": dict(
        speed=lambda t: 8.0 + 0 * t, curvature=lambda t: 0.01 + 0 * t, future_shift=(30, 0.12)
    ),
    "noisy_pitch_roll": dict(
        speed=lambda t: 12.0 + 0 * t,
        curvature=lambda t: -0.015 + 0 * t,
        noise=(0.02, 1e-3),
        pitch_roll=(0.02, -0.01),
    ),
}


def _rotation(yaw: torch.Tensor, pitch: float, roll: float) -> torch.Tensor:
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    rz = torch.stack([cy, -sy, zero, sy, cy, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=yaw.dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=yaw.dtype)
    return rz @ ry @ rx


def _scenario(
    seed: int,
    speed,
    curvature,
    history_shift: float = 0.0,
    future_shift: tuple[int, float] | None = None,
    noise: tuple[float, float] = (0.0, 0.0),
    pitch_roll: tuple[float, float] = (0.0, 0.0),
    substeps: int = 10,
) -> tuple[torch.Tensor, ...]:
    """Rear-axle poses in the t0 rig frame; body yaw = course + rear-axle sideslip."""
    h = DT / substeps
    t = torch.arange(-(HISTORY_STEPS - 1) * substeps, FUTURE_STEPS * substeps + 1) * h
    t = t.double()
    v, kappa = speed(t).double(), curvature(t).double()
    course = torch.cat([t.new_zeros(1), torch.cumsum(kappa * v * h, 0)[:-1]])
    x = torch.cat([t.new_zeros(1), torch.cumsum(v * torch.cos(course) * h, 0)[:-1]])
    y = torch.cat([t.new_zeros(1), torch.cumsum(v * torch.sin(course) * h, 0)[:-1]])
    lateral_accel = kappa * v * v.abs()
    yaw = course + torch.where(v.abs() > 0.1, 0.4e-3 + 2.7e-3 * lateral_accel, 0.0)
    x, y, yaw = x[::substeps], y[::substeps], yaw[::substeps]
    index = torch.arange(len(x), dtype=torch.float64)
    x = x + noise[0] * _hash_noise(index, seed)
    y = y + noise[0] * _hash_noise(index, seed + 0.5)
    yaw = yaw + noise[1] * _hash_noise(index, seed + 0.25)
    # Express in the frame of the t0 pose.
    t0 = HISTORY_STEPS - 1
    dx, dy = x - x[t0], y - y[t0]
    c, s = math.cos(yaw[t0]), math.sin(yaw[t0])
    xy = torch.stack([c * dx + s * dy, -s * dx + c * dy], -1)
    yaw = yaw - yaw[t0]
    if history_shift:
        xy[t0 - 1, 1] += history_shift
    if future_shift is not None:
        xy[t0 + 1 + future_shift[0], 1] += future_shift[1]
    xyz = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
    rot = _rotation(yaw, *pitch_roll)
    return xyz[:HISTORY_STEPS], rot[:HISTORY_STEPS], xyz[HISTORY_STEPS:], rot[HISTORY_STEPS:]


def scenario_batch(dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """All scenarios stacked, plus a deterministic sampled action for each, in ``dtype``."""
    parts = [_scenario(i, **kwargs) for i, kwargs in enumerate(SCENARIOS.values())]
    hist_xyz, hist_rot, fut_xyz, fut_rot = (torch.stack(p) for p in zip(*parts, strict=True))
    k = torch.arange(FUTURE_STEPS, dtype=torch.float32)
    b = torch.arange(len(SCENARIOS), dtype=torch.float32)[:, None]
    sampled = torch.stack([1.5 * torch.sin(0.1 * k + b), 2.0 * torch.cos(0.07 * k + 0.5 * b)], -1)
    batch = {
        "hist_xyz": hist_xyz,
        "hist_rot": hist_rot,
        "fut_xyz": fut_xyz,
        "fut_rot": fut_rot,
        "sampled_action": sampled,
    }
    return {key: value.to(dtype) for key, value in batch.items()}


def _yaw(rot: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def compute_outputs(space, inputs: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Encode the futures, then decode the encodings and the sampled actions as inference does."""
    hist = inputs["hist_xyz"], inputs["hist_rot"]
    t0 = space.estimate_t0_states(*hist)
    action = space.traj_to_action(*hist, inputs["fut_xyz"], inputs["fut_rot"])
    outputs = {f"{prefix}.t0_{key}": value for key, value in t0.items()}
    outputs[f"{prefix}.action"] = action
    for name, decoded_action in (("roundtrip", action), ("sampled", inputs["sampled_action"])):
        xyz, rot = space.action_to_traj(decoded_action, *hist)
        outputs[f"{prefix}.{name}_xyz"] = xyz
        outputs[f"{prefix}.{name}_yaw"] = _yaw(rot)
    return outputs


def _angle_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, msg: str) -> None:
    difference = torch.remainder(actual - expected + math.pi, 2 * math.pi) - math.pi
    torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=atol, msg=msg)


@pytest.fixture(scope="module")
def reference() -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(DATA)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("prefix", ["course", "yaw"])
def test_encode_and_decode_match_reference(reference, prefix, dtype):
    from alpamayo2_edge.action_space import (
        UnicycleAccelCurvatureActionSpace,
        UnicycleAccelCurvatureCourseActionSpace,
    )

    space = (
        UnicycleAccelCurvatureCourseActionSpace(**COURSE_CFG)
        if prefix == "course"
        else UnicycleAccelCurvatureActionSpace(**YAW_CFG)
    )
    actual = compute_outputs(space, scenario_batch(dtype), prefix)
    expected = {key: value for key, value in reference.items() if key.startswith(f"{prefix}.")}
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        kind = key.rsplit(".", 1)[1]
        kind = "t0" if kind.startswith("t0_") else kind.rsplit("_", 1)[-1]
        atol = TOLERANCES[dtype][kind]
        if kind == "yaw":
            _angle_close(actual[key].double(), value, atol=atol, msg=key)
        else:
            # Normalized actions, positions [m], t0 speed [m/s] and course [rad].
            torch.testing.assert_close(actual[key].double(), value, rtol=0, atol=atol, msg=key)


def test_course_and_yaw_spaces_differ_on_these_scenarios(reference):
    # Guards against a parity check that would also pass with the yaw space under the course name.
    moved = (reference["course.roundtrip_xyz"] - reference["yaw.roundtrip_xyz"]).norm(dim=-1)
    assert moved.amax() > 0.05


def test_decoding_without_t0_states_uses_full_history_course():
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace

    space = UnicycleAccelCurvatureCourseActionSpace(**COURSE_CFG)
    inputs = scenario_batch()
    hist = inputs["hist_xyz"], inputs["hist_rot"]
    t0_states = space.estimate_t0_states(*hist)
    assert t0_states["theta"].abs().amax() > 1e-3
    implicit = space.action_to_traj(inputs["sampled_action"], *hist)
    explicit = space.action_to_traj(inputs["sampled_action"], *hist, t0_states=t0_states)
    for actual, expected in zip(implicit, explicit, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_course_space_needs_two_history_poses_and_its_own_t0_states():
    from alpamayo2_edge.action_space import (
        UnicycleAccelCurvatureActionSpace,
        UnicycleAccelCurvatureCourseActionSpace,
    )

    space = UnicycleAccelCurvatureCourseActionSpace(**COURSE_CFG)
    inputs = scenario_batch()
    with pytest.raises(ValueError, match="at least 2 history poses"):
        space.estimate_t0_states(inputs["hist_xyz"][:, -1:], inputs["hist_rot"][:, -1:])
    yaw_states = UnicycleAccelCurvatureActionSpace(**YAW_CFG).estimate_t0_states(
        inputs["hist_xyz"], inputs["hist_rot"]
    )
    with pytest.raises(ValueError, match=r"t0_states\['theta'\]"):
        space.action_to_traj(
            inputs["sampled_action"], inputs["hist_xyz"], inputs["hist_rot"], yaw_states
        )


def test_slip_gradient_changes_only_decoded_rotations():
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace

    inputs = scenario_batch()
    decoded = [
        UnicycleAccelCurvatureCourseActionSpace(
            **{**COURSE_CFG, "slip_gradient": slip}
        ).action_to_traj(inputs["sampled_action"], inputs["hist_xyz"], inputs["hist_rot"])
        for slip in (0.0, COURSE_CFG["slip_gradient"])
    ]
    torch.testing.assert_close(decoded[0][0], decoded[1][0], rtol=0, atol=0)
    assert (_yaw(decoded[0][1]) - _yaw(decoded[1][1])).abs().amax() > 1e-3
