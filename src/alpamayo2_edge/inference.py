# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the trained vision-language backbone and trajectory expert on local samples."""

import argparse
import json
import time
from pathlib import Path

import torch


def validate_predictions(xyz, rot, num_sets: int, num_samples: int) -> None:
    expected = (1, num_sets, num_samples, 64)
    if tuple(xyz.shape) != (*expected, 3) or tuple(rot.shape) != (*expected, 3, 3):
        raise ValueError(
            f"Unexpected trajectory shape: xyz={tuple(xyz.shape)}, rot={tuple(rot.shape)}"
        )
    if not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
        raise ValueError("Model produced nonfinite trajectory values")


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def run_sample(model, sample_path: Path, output: Path, settings: dict, device: str) -> dict:
    from safetensors.torch import save_file

    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.models.utils import fuse_traj_tokens
    from alpamayo2_edge.sample import load_sample

    data = load_sample(sample_path)
    start = time.perf_counter()
    prepared = prepare_model_inputs(data, model.config, model.tokenizer)
    packed = {
        f"tokenized_data.{key}": value.cpu().contiguous()
        for key, value in prepared["tokenized_data"].items()
    }
    histories = {key: prepared[key] for key in ("ego_history_xyz", "ego_history_rot")}
    packed["tokenized_data.input_ids"] = (
        fuse_traj_tokens(
            model.history_traj_tokenizer,
            model.future_traj_tokenizer,
            prepared["tokenized_data"]["input_ids"],
            histories,
            model.config.traj_ids,
        )
        .cpu()
        .contiguous()
    )
    packed.update(
        {f"traj_data.{key}": value.cpu().contiguous() for key, value in histories.items()}
    )
    prepared = to_device(prepared, device=device)
    torch.manual_seed(settings["seed"])
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=model.dtype,
            enabled=model.dtype != torch.float32,
        ),
    ):
        xyz, rot, _, extra = model.sample_trajectories_from_data(
            prepared,
            num_traj_sets=settings["num_traj_sets"],
            num_traj_samples=settings["num_traj_samples"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            temperature=settings["temperature"],
            max_generation_length=settings["max_generation_length"],
            diffusion_kwargs={**settings["diffusion_kwargs"], "generator": generator},
            return_extra=True,
        )
    validate_predictions(xyz, rot, settings["num_traj_sets"], settings["num_traj_samples"])
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "pred_xyz": xyz.detach().float().cpu().contiguous(),
        "pred_rot": rot.detach().float().cpu().contiguous(),
    }
    if isinstance(extra.get("generated_ids"), torch.Tensor):
        tensors["generated_ids"] = extra["generated_ids"].detach().cpu().contiguous()
    save_file(tensors, str(output / "output.safetensors"))
    save_file(packed, str(output / "packed.safetensors"))
    prompt = model.tokenizer.decode(
        packed["tokenized_data.input_ids"][0], skip_special_tokens=False
    )
    (output / "prompt.txt").write_text(prompt)
    record = {
        "sample_id": sample_path.name,
        "pred_xyz_shape": list(xyz.shape),
        "pred_rot_shape": list(rot.shape),
        "finite": True,
        "elapsed_seconds": elapsed,
        "text": _json_value(
            {key: value for key, value in extra.items() if key in ("cot", "answer")}
        ),
    }
    (output / "output.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Local bundle containing model/ and samples/")
    source.add_argument("--model-id", help="Local model directory or Hugging Face repository ID")
    parser.add_argument(
        "--subfolder", default="", help="HF model subfolder, including model/ if needed"
    )
    parser.add_argument("--revision", help="Immutable HF revision for reproducible model downloads")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sample", type=Path, help="Raw sample directory for --model-id")
    parser.add_argument(
        "--sample-id", action="append", help="Bundle sample ID (repeatable; default sample_000)"
    )
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument("--num-traj-sets", type=int)
    parser.add_argument("--num-traj-samples", type=int)
    parser.add_argument("--max-generation-length", type=int)
    parser.add_argument("--diffusion-steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.model_id and args.sample is None:
        parser.error("--model-id requires --sample")
    if args.sample_id and not args.bundle:
        parser.error("--sample-id requires --bundle")
    if args.sample is not None and args.bundle:
        parser.error("--sample cannot be combined with --bundle")
    if args.all_samples and (not args.bundle or args.sample_id):
        parser.error("--all-samples requires --bundle and cannot be combined with --sample-id")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; run on a GPU host or explicitly select --device cpu --dtype float32"
        )
    settings = dict(
        seed=43,
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_sets=1,
        num_traj_samples=1,
        max_generation_length=256,
        diffusion_kwargs={"inference_step": 10},
    )
    if args.bundle:
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        settings.update(
            {key: value for key, value in manifest["inference"].items() if key in settings}
        )
        entries = {entry["id"]: args.bundle / entry["path"] for entry in manifest["samples"]}
        ids = list(entries) if args.all_samples else args.sample_id or ["sample_000"]
        unknown = set(ids) - entries.keys()
        if unknown:
            parser.error(f"Unknown sample IDs: {sorted(unknown)}")
        samples = [entries[key] for key in ids]
    else:
        samples = [args.sample]
    for key in ("seed", "num_traj_sets", "num_traj_samples", "max_generation_length"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.diffusion_steps is not None:
        settings["diffusion_kwargs"] = {"inference_step": args.diffusion_steps}
    if (
        any(
            settings[key] <= 0
            for key in ("num_traj_sets", "num_traj_samples", "max_generation_length")
        )
        or settings["diffusion_kwargs"]["inference_step"] <= 0
    ):
        parser.error("Sampling counts, generation length and diffusion steps must be positive")
    import transformers

    from alpamayo2_edge.checkpoint import load_model

    model, loading_info = load_model(
        str(args.bundle or args.model_id),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        subfolder=args.subfolder,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    records = [
        run_sample(model, sample, args.output / sample.name, settings, args.device)
        for sample in samples
    ]
    report = {
        "model_path": model.config._name_or_path,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(args.device)
        if torch.device(args.device).type == "cuda"
        else "cpu",
        "attention_backend": args.attn_implementation,
        "settings": settings,
        "loading_info": loading_info,
        "samples": records,
    }
    (args.output / "report.json").write_text(json.dumps(_json_value(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
