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
    "args",
    [
        ["--model-id", "dummy", "--sample", "sample_000", "--sample-id", "sample_999"],
        ["--bundle", "bundle", "--sample", "sample_000"],
    ],
)
def test_cli_rejects_conflicting_sample_sources(args, capsys):
    from alpamayo2_edge.inference import main

    with pytest.raises(SystemExit) as error:
        main([*args, "--output", "output", "--device", "cpu"])
    assert error.value.code == 2
    assert "sample" in capsys.readouterr().err


def test_report_records_selected_cuda_device(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from alpamayo2_edge import checkpoint, inference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: f"GPU at {device}")
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="model"))
    monkeypatch.setattr(
        checkpoint, "load_model", lambda *args, **kwargs: (model, {"missing_keys": set()})
    )

    def run_sample(*args):
        tmp_path.mkdir(exist_ok=True)
        return {"finite": True}

    monkeypatch.setattr(inference, "run_sample", run_sample)
    inference.main(
        [
            "--model-id",
            "model",
            "--sample",
            "sample_000",
            "--device",
            "cuda:1",
            "--output",
            str(tmp_path),
        ]
    )
    assert json.loads((tmp_path / "report.json").read_text())["device"] == "GPU at cuda:1"
