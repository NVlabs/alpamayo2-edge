# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Alpamayo 2 Edge to ExecuTorch programs made of TensorRT engines.

vision.pte  vision_<H>x<W>, one method per camera patch grid
text.pte    prefill, decode and expert_step, which share one KV cache that their
            engines update in place

TensorRT builds the engines for the GPU it runs on, so export on the target device.

Usage:
  python examples/executorch_tensorrt/export.py --model-id "$EDGE_MODEL"
"""

from __future__ import annotations

import argparse
import collections
import os
import tempfile

import torch
import torch_tensorrt
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from host import HostPipeline, synthetic_sample
from model import NUM_ACTION_TOKENS, build_methods
from torch.export import Dim
from torch_tensorrt.executorch import check_zero_copy_kv, export, zero_copy_backend_config

from alpamayo2_edge.checkpoint import load_model


def compile_method(name, module, args, dynamic_shapes=None, inputs=None):
    """One method as an ExportedProgram holding a single TensorRT engine."""
    print(f"Compiling {name}...", flush=True)
    program = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    gm = torch_tensorrt.dynamo.compile(
        program, inputs=inputs or list(args), min_block_size=1, truncate_double=True
    )
    engines = [n.target for n in gm.graph.nodes if n.op == "call_module"]
    # Anything left out of TensorRT would run on ExecuTorch's CPU kernels.
    if len(engines) != 1 or not str(engines[0]).startswith("_run_on_acc"):
        raise RuntimeError(f"{name}: expected one TensorRT engine, got {engines}")
    # export() takes several methods only as programs; this is how it converts one module.
    program = torch_tensorrt.dynamo.export(
        gm, arg_inputs=list(args), dynamic_shapes=dynamic_shapes, use_legacy_exporter=True
    )
    del gm
    torch.cuda.empty_cache()
    return program


def write(program, path: str) -> None:
    with open(path, "wb") as f:
        program.write_to_file(f)
    print(f"Built {os.path.basename(path)} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-id", required=True, help="Local model directory or Hugging Face repository ID"
    )
    parser.add_argument("--output-dir", default="outputs/et_trt")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    host = HostPipeline(args.model_id)
    # The patch grids follow from each camera's frame size.
    prompt = host.prepare(synthetic_sample())
    length = prompt.tokens.shape[1]
    max_seq_len = args.max_seq_len
    needed = length + host.max_new_tokens + NUM_ACTION_TOKENS
    if needed > max_seq_len:
        raise ValueError(f"--max-seq-len {max_seq_len} is too short: the sample needs {needed}")
    images_per_grid = collections.Counter(prompt.grids)
    grids = {f"vision_{h}x{w}": (h, w) for h, w in images_per_grid}

    model = load_model(args.model_id, device="cuda", dtype="bfloat16")[0]
    hidden = model.vlm.config.text_config.hidden_size
    vocab = model.vlm.lm_head.weight.shape[0]
    patch_dim = prompt.patches[0].shape[-1]
    methods = build_methods(model, max_seq_len, host.blocked_token_ids(), grids)
    methods = {name: m.cuda() for name, m in methods.items()}
    num_steps = model.expert.diffusion.num_inference_steps
    del model
    os.makedirs(args.output_dir, exist_ok=True)
    # Both programs replace the old pair only after both are built, so a failed build
    # keeps the old pair.
    staging = tempfile.TemporaryDirectory(dir=args.output_dir)

    with torch.no_grad():
        vision = {
            name: compile_method(
                name,
                methods[name],
                (torch.zeros(images_per_grid[hw], hw[0] * hw[1], patch_dim, device="cuda"),),
            )
            for name, hw in grids.items()
        }
        write(export(vision).to_executorch(), os.path.join(staging.name, "vision.pte"))
        del vision

        def i64(value: int) -> torch.Tensor:
            return torch.tensor([value], dtype=torch.int64, device="cuda")

        temperature = torch.ones(1, device="cuda")
        uniform = torch.rand(1, vocab, device="cuda")
        # Room for one reasoning token and the action tokens after the prompt. It also
        # keeps cache[:, :, :T] from making torch.export fix T at max_seq_len.
        max_prompt = max_seq_len - 1 - NUM_ACTION_TOKENS
        prompt_len = Dim("prompt_len", min=2, max=max_prompt)
        prefill_args = (
            torch.zeros(1, length, dtype=torch.int64, device="cuda"),
            torch.zeros(1, length, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(3, 1, length, dtype=torch.int64, device="cuda"),
            temperature,
            uniform,
        )
        dynamic_dims = {0: 1, 1: 1, 2: 2}
        prefill_inputs = []
        for i, a in enumerate(prefill_args):
            low, high = list(a.shape), list(a.shape)
            if i in dynamic_dims:
                low[dynamic_dims[i]], high[dynamic_dims[i]] = 2, max_prompt
            prefill_inputs.append(
                torch_tensorrt.Input(
                    min_shape=low, opt_shape=list(a.shape), max_shape=high, dtype=a.dtype
                )
            )
        text = {
            "prefill": compile_method(
                "prefill",
                methods["prefill"],
                prefill_args,
                dynamic_shapes=({1: prompt_len}, {1: prompt_len}, {2: prompt_len}, None, None),
                inputs=prefill_inputs,
            ),
            "decode": compile_method(
                "decode",
                methods["decode"],
                (
                    torch.zeros(1, 1, dtype=torch.int64, device="cuda"),
                    i64(0),
                    i64(0),
                    temperature,
                    uniform,
                ),
            ),
            "expert_step": compile_method(
                "expert_step",
                methods["expert_step"],
                (
                    torch.zeros(1, NUM_ACTION_TOKENS, 2, device="cuda"),
                    torch.zeros(1, device="cuda"),
                    torch.full((1,), 1.0 / num_steps, device="cuda"),
                    i64(0),
                    i64(0),
                ),
            ),
        }
    del methods
    torch.cuda.empty_cache()

    print("Lowering text.pte...", flush=True)
    edge = export(
        text,
        zero_copy_kv=True,
        constant_methods={
            "get_max_seq_len": max_seq_len,
            "get_anchor_token": host.anchor_token,
            "get_num_steps": num_steps,
            "get_vocab_size": vocab,
        },
    )
    # The cache is planned once in GPU memory and shared by all three methods.
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True, alignment=256),
        enable_non_cpu_memory_planning=True,
    )
    program = edge.to_executorch(zero_copy_backend_config(config))
    check_zero_copy_kv(program)
    write(program, os.path.join(staging.name, "text.pte"))
    for name in ("vision.pte", "text.pte"):
        os.replace(os.path.join(staging.name, name), os.path.join(args.output_dir, name))
    staging.cleanup()
    print(f"Saved vision.pte and text.pte in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
