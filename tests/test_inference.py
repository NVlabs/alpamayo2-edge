import pytest
import torch

from alpamayo2_edge.inference import validate_predictions


def test_nonfinite_or_wrong_trajectory_is_rejected():
    xyz = torch.zeros(1, 2, 3, 64, 3)
    rot = torch.eye(3).expand(1, 2, 3, 64, 3, 3).clone()
    validate_predictions(xyz, rot, 2, 3)
    with pytest.raises(ValueError, match="shape"):
        validate_predictions(xyz[:, :, :, :63], rot, 2, 3)
    xyz[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_predictions(xyz, rot, 2, 3)


def test_hf5_loading_diagnostics_serialize_sets():
    import json

    from alpamayo2_edge.inference import _json_value

    assert json.loads(json.dumps(_json_value({"missing_keys": set()}))) == {"missing_keys": []}


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--sample", "0"],
        ["--bundle", "bundle"],
        ["--sample-id", "sample_000"],
        ["--all-samples"],
    ],
)
def test_cli_rejects_removed_local_sample_options(removed_args, capsys):
    from alpamayo2_edge.inference import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--model-id",
                "model",
                "--manifest",
                "clips.json",
                "--output",
                "output",
                "--device",
                "cpu",
                *removed_args,
            ]
        )
    assert error.value.code == 2
    assert f"unrecognized arguments: {removed_args[0]}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["--model-id", "model"],
        ["--manifest", "clips.json"],
        ["--model-id", "model", "--clip-id", "clip", "--manifest", "clips.json"],
    ],
)
def test_cli_requires_model_and_one_physical_ai_source(args):
    from alpamayo2_edge.inference import main

    with pytest.raises(SystemExit) as error:
        main([*args, "--output", "output", "--device", "cpu"])
    assert error.value.code == 2


@pytest.mark.parametrize("use_manifest", [False, True])
def test_cli_runs_physical_ai_clip_and_records_settings(tmp_path, monkeypatch, use_manifest):
    import json
    from types import SimpleNamespace

    from alpamayo2_edge import checkpoint, inference, load_physical_aiavdataset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: f"GPU at {device}")
    _mock_dataset_versions(monkeypatch)
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="model"))
    monkeypatch.setattr(
        checkpoint, "load_model", lambda *args, **kwargs: (model, {"missing_keys": set()})
    )

    clip = {
        "clip_id": "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
        "t0_us": 4_000_000,
        "nav_text": "Turn right in 30m",
    }
    data = {"clip_id": clip["clip_id"]}
    loaded_clips = []

    def load_clip(**kwargs):
        loaded_clips.append(kwargs)
        return data

    monkeypatch.setattr(load_physical_aiavdataset, "load_physical_aiavdataset", load_clip)

    def run_data(actual_model, actual_data, sample_id, output, settings, device):
        assert actual_model is model
        assert actual_data is data
        assert output == tmp_path / sample_id
        assert device == "cuda:1"
        output.mkdir(parents=True)
        return {"sample_id": sample_id, "finite": True}

    monkeypatch.setattr(inference, "run_data", run_data)
    if use_manifest:
        path = tmp_path / "clips.json"
        path.write_text(json.dumps({"samples": [{**clip, "nav_text": "Old route"}, clip]}))
        source_args = ["--manifest", str(path), "--sample-index", "1"]
    else:
        source_args = [
            "--clip-id",
            clip["clip_id"],
            "--t0-us",
            str(clip["t0_us"]),
            "--nav-text",
            clip["nav_text"],
        ]
    inference.main(
        [
            "--model-id",
            "model",
            *source_args,
            "--num-traj-samples",
            "6",
            "--seed",
            "7",
            "--diffusion-steps",
            "4",
            "--device",
            "cuda:1",
            "--output",
            str(tmp_path),
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert loaded_clips == [clip]
    assert report["device"] == "GPU at cuda:1"
    assert report["settings"] == {
        "seed": 7,
        "top_p": 0.98,
        "top_k": None,
        "temperature": 0.6,
        "num_traj_sets": 1,
        "num_traj_samples": 6,
        "max_generation_length": 256,
        "diffusion_kwargs": {"inference_step": 4},
    }
    assert report["samples"] == [{"sample_id": f"{clip['clip_id']}_4000000", "finite": True}]
    assert report["dataset_runtime"]["physical-ai-av"] == "test-physical-ai-av"


def test_clip_manifest_uses_paired_navigation_and_explicit_override(tmp_path):
    import json

    from alpamayo2_edge.inference import load_clip_manifest

    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "clip_id": "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
                        "t0_us": 4000000,
                        "nav_text": "Turn right in 30m",
                    },
                    {"clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26", "t0_us": 5100000},
                ]
            }
        )
    )
    assert load_clip_manifest(path, 0) == {
        "clip_id": "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
        "t0_us": 4000000,
        "nav_text": "Turn right in 30m",
    }
    # Super's manifest stores clip/time only; its demo supplies navigation separately.
    assert (
        load_clip_manifest(path, 1, nav_text="Continue straight")["nav_text"] == "Continue straight"
    )
    with pytest.raises(ValueError, match="nav"):
        load_clip_manifest(path, 1)
    with pytest.raises(ValueError, match="index"):
        load_clip_manifest(path, -1)
    with pytest.raises(ValueError, match="index"):
        load_clip_manifest(path, 2)


@pytest.mark.parametrize(
    "field,value", [("clip_id", "../bad"), ("t0_us", 1.2), ("t0_us", True), ("nav_text", " ")]
)
def test_clip_manifest_rejects_invalid_inputs_before_download(tmp_path, field, value):
    import json

    from alpamayo2_edge.inference import load_clip_manifest

    sample = {
        "clip_id": "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
        "t0_us": 4000000,
        "nav_text": "Turn right in 30m",
        field: value,
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"samples": [sample]}))
    with pytest.raises(ValueError, match=field):
        load_clip_manifest(path, 0)


def _mock_dataset_versions(monkeypatch):
    import importlib.metadata

    original_version = importlib.metadata.version
    dataset_packages = {"physical-ai-av", "av", "pandas", "pyarrow"}
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: f"test-{name}" if name in dataset_packages else original_version(name),
    )


def _clip_cli_args(output):
    return [
        "--model-id",
        "model",
        "--clip-id",
        "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
        "--nav-text",
        "Continue straight",
        "--output",
        str(output),
    ]


def _fake_cli_model(monkeypatch, flash_attn=True):
    from types import SimpleNamespace

    from alpamayo2_edge import accel, checkpoint, inference, load_physical_aiavdataset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU")
    _mock_dataset_versions(monkeypatch)
    kernel = "fa2" if flash_attn else None
    kernels = {"vision": "sdpa", "prefill": "sdpa", "decode": kernel, "expert": kernel}
    monkeypatch.setattr(accel, "resolve_kernels", lambda attention, device: kernels)
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="model"))
    monkeypatch.setattr(checkpoint, "load_model", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(
        load_physical_aiavdataset, "load_physical_aiavdataset", lambda **clip: {"clip": clip}
    )
    monkeypatch.setattr(inference, "run_data", lambda *args: {"finite": True})
    return model


def test_report_records_linear_patch_embedding(tmp_path, monkeypatch):
    import json

    from alpamayo2_edge import accel, inference

    model = _fake_cli_model(monkeypatch)
    calls = []
    monkeypatch.setattr(accel, "use_linear_patch_embedding", calls.append)
    args = _clip_cli_args(tmp_path)
    inference.main(args)
    assert "patch_embedding" not in json.loads((tmp_path / "report.json").read_text())
    inference.main([*args, "--patch-embed", "linear"])
    assert json.loads((tmp_path / "report.json").read_text())["patch_embedding"] == "linear"
    assert calls == [model]


def test_cli_acceleration_requires_cuda_and_bfloat16(capsys):
    from alpamayo2_edge.inference import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                *_clip_cli_args("output"),
                *("--device", "cpu", "--dtype", "float32", "--accel", "tuned"),
            ]
        )
    assert error.value.code == 2
    assert "--accel requires" in capsys.readouterr().err


def test_report_records_acceleration(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from alpamayo2_edge import accel, inference

    model = _fake_cli_model(monkeypatch)
    calls = []

    def enable_acceleration(model, config):
        calls.append(config)
        return SimpleNamespace(describe=lambda: {"parts": ["vision"], "fallbacks": {}})

    monkeypatch.setattr(accel, "enable_acceleration", enable_acceleration)
    monkeypatch.setattr(inference, "warm_up", lambda *args: calls.append(args))
    inference.main(
        [
            *_clip_cli_args(tmp_path),
            *("--accel", "compile", "--accel-attention", "sdpa", "--accel-warmup", "2"),
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["acceleration"]["parts"] == ["vision"]
    assert report["acceleration"]["warmup_runs"] == 2
    assert (calls[0].mode, calls[0].attention) == ("compile", "sdpa")
    assert calls[1][0] is model
    assert calls[1][1] == {
        "clip": {
            "clip_id": "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
            "t0_us": 5_100_000,
            "nav_text": "Continue straight",
        }
    }
    assert calls[1][-1] == 2


def test_cli_acceleration_without_flash_attention_needs_allow_partial(
    tmp_path, monkeypatch, capsys
):
    import json
    from types import SimpleNamespace

    from alpamayo2_edge import accel, checkpoint, inference

    _fake_cli_model(monkeypatch, flash_attn=False)
    loads = []
    monkeypatch.setattr(checkpoint, "load_model", lambda *args, **kwargs: loads.append(1))
    args = _clip_cli_args(tmp_path)
    with pytest.raises(SystemExit) as error:
        inference.main([*args, "--accel", "tuned"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "flash_attn" in message and "--accel-allow-partial" in message
    assert not loads  # Fails before loading the checkpoint.
    with pytest.raises(SystemExit):
        inference.main([*args, "--accel-allow-partial"])
    assert "requires --accel" in capsys.readouterr().err

    model = _fake_cli_model(monkeypatch, flash_attn=False)
    description = {"parts": ["vision"], "skipped": {"decode": "requires flash_attn"}}
    monkeypatch.setattr(
        accel,
        "enable_acceleration",
        lambda model, config: SimpleNamespace(describe=lambda: description),
    )
    monkeypatch.setattr(inference, "warm_up", lambda *args: None)
    inference.main([*args, "--accel", "tuned", "--accel-allow-partial"])
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["acceleration"]["skipped"] == {"decode": "requires flash_attn"}
    assert model.config._name_or_path == "model"


def test_cli_rejects_unavailable_attention_before_loading(monkeypatch, capsys):
    from alpamayo2_edge import accel, inference

    _fake_cli_model(monkeypatch)

    def resolve(attention, device):
        raise ValueError("attention='fa3' requires a Hopper (sm_90) GPU")

    monkeypatch.setattr(accel, "resolve_kernels", resolve)
    with pytest.raises(SystemExit):
        inference.main(
            [
                *_clip_cli_args("output"),
                *("--accel", "tuned", "--accel-attention", "fa3"),
            ]
        )
    assert "Hopper" in capsys.readouterr().err
