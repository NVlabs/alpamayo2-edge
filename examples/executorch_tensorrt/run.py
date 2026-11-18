# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the exported Alpamayo 2 Edge programs with the ExecuTorch runtime.

The flow for one driving sample:

  camera patches -> vision_<H>x<W> per patch grid -> image embeddings
  prompt tokens + image embeddings -> prefill -> first token
  decode, one token per call, until the trajectory anchor token is in the cache
  expert_step, once per flow-matching step -> action -> 64 future poses

Usage:
  python examples/executorch_tensorrt/run.py --model-id "$EDGE_MODEL" --check
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401  (registers the TensorRT backend)
from executorch.runtime import Runtime
from host import HostPipeline, Prompt, synthetic_sample
from model import NUM_ACTION_TOKENS

# Trajectory bound for --check. On 64 synthetic inputs, the PyTorch model against itself
# with only the attention kernel changed moves the path by up to 20 cm (median 3.5 cm).
MAX_PATH_ERROR_M = 0.25


class Runner:
    """Drives the vision and text programs for one sample at a time."""

    def __init__(self, pte_dir: str, cuda_graphs: bool) -> None:
        runtime = Runtime.get()
        registry = runtime.backend_registry
        registry.set_option("TensorRTBackend", {"use_cuda_graphs": False})
        vision = runtime.load_program(os.path.join(pte_dir, "vision.pte"))
        self.vision = {name: vision.load_method(name) for name in vision.method_names}
        self.images_per_grid = {
            name: vision.metadata(name).input_tensor_meta(0).sizes()[0]
            for name in vision.method_names
        }
        text = runtime.load_program(os.path.join(pte_dir, "text.pte"))
        # Loading a method resets the shared cache, so every method loads before any runs.
        # prefill changes its length on every sample, which a CUDA graph cannot follow.
        self.text = {}
        for name in ("prefill", "decode", "expert_step"):
            registry.set_option(
                "TensorRTBackend", {"use_cuda_graphs": cuda_graphs and name != "prefill"}
            )
            self.text[name] = text.load_method(name)

        def constant(name: str) -> int:
            return int(text.load_method(name).execute([])[0])

        self.max_seq_len = constant("get_max_seq_len")
        self.anchor_token = constant("get_anchor_token")
        self.num_steps = constant("get_num_steps")
        self.vocab_size = constant("get_vocab_size")
        # Greedy sampling ignores the noise, so it is not drawn inside the timed stages.
        self.no_noise = torch.zeros(1, self.vocab_size)
        self.timings: dict[str, float] = {}
        # Kept alive with their methods.
        self.programs = (vision, text)

    def encode_images(self, prompt: Prompt) -> torch.Tensor:
        """Image embeddings in prompt order, [num_image_tokens, hidden]."""
        per_image = [torch.empty(0)] * len(prompt.grids)
        for grid in dict.fromkeys(prompt.grids):
            picked = [i for i, g in enumerate(prompt.grids) if g == grid]
            name = f"vision_{grid[0]}x{grid[1]}"
            if self.images_per_grid.get(name) != len(picked):
                exported = ", ".join(f"{k} ({n} images)" for k, n in self.images_per_grid.items())
                raise ValueError(
                    f"The frames give {name} ({len(picked)} images), but vision.pte has {exported}. "
                    "Export again with frames of this size, set in the synthetic_sample() call in "
                    "export.py."
                )
            batch = torch.stack([prompt.patches[i] for i in picked])
            out = self.vision[name].execute([batch])[0]
            for row, i in enumerate(picked):
                per_image[i] = out[row]
        return torch.cat(per_image)

    def run(
        self,
        host: HostPipeline,
        prompt: Prompt,
        temperature: float,
        noise: torch.Tensor,
        generator: torch.Generator,
    ) -> dict:
        """Reasoning tokens and the [1, 64, 2] action for one prepared sample."""
        tokens = prompt.tokens
        length = tokens.shape[1]
        if length + 1 + NUM_ACTION_TOKENS > self.max_seq_len:
            raise ValueError(
                "The prompt does not fit in the KV cache; export a longer --max-seq-len"
            )
        temp = torch.tensor([temperature], dtype=torch.float32)

        def uniform() -> torch.Tensor:
            if temperature == 0:
                return self.no_noise
            return torch.rand(1, self.vocab_size, generator=generator)

        t0 = time.perf_counter()
        image_embeds = self.encode_images(prompt)
        t1 = time.perf_counter()
        images = torch.zeros(1, length, image_embeds.shape[-1], dtype=image_embeds.dtype)
        images[0, tokens[0] == host.config.vlm_config.image_token_id] = image_embeds
        args = [tokens, images, prompt.positions, temp, uniform()]
        generated = [int(self.text["prefill"].execute(args)[0])]
        t2 = time.perf_counter()
        pos = length
        delta = torch.tensor([prompt.rope_delta])
        # Decode writes the last token into the cache, so the anchor token is fed too.
        while True:
            token = torch.tensor([[generated[-1]]])
            next_token = self.text["decode"].execute(
                [token, torch.tensor([pos]), delta, temp, uniform()]
            )[0]
            pos += 1
            if generated[-1] == self.anchor_token:
                break
            if len(generated) == host.max_new_tokens:
                raise RuntimeError("No trajectory anchor token within the length limit")
            if pos + 1 + NUM_ACTION_TOKENS > self.max_seq_len:
                raise RuntimeError("The KV cache filled up; export a longer --max-seq-len")
            generated.append(int(next_token))
        t3 = time.perf_counter()
        prefix_len = torch.tensor([pos])
        timesteps = torch.linspace(0.0, 1.0, self.num_steps + 1)
        x = noise
        for s in range(self.num_steps):
            t, dt = timesteps[s : s + 1], timesteps[s + 1 : s + 2] - timesteps[s : s + 1]
            x = self.text["expert_step"].execute([x, t, dt, prefix_len, delta])[0]
        t4 = time.perf_counter()
        self.timings = {
            "vision": t1 - t0,
            "prefill": t2 - t1,
            "decode": t3 - t2,
            "expert": t4 - t3,
            "total": t4 - t0,
        }
        return {"generated": generated, "action": x}


def upstream_reference(model_id: str, sample: dict, seed: int) -> tuple[list[int], torch.Tensor]:
    """Reasoning tokens and the [64, 3] path from the PyTorch model with top_k=1.

    The PyTorch model always samples, so this is greedy except at exactly tied scores.
    """
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    model = load_model(model_id, device="cuda", dtype="bfloat16")[0]
    data = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device="cuda")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, _, _, extra = model.sample_trajectories_from_data(
            data, top_k=1, diffusion_kwargs={"generator": generator}, return_extra=True
        )
    tokens = extra["generated_ids"][0]
    return tokens[tokens >= 0].tolist(), xyz[0, 0, 0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-id", required=True, help="Local model directory or Hugging Face repository ID"
    )
    parser.add_argument("--pte-dir", default="outputs/et_trt")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="PhysicalAI-AV example manifest (needs the dataset extra); default: a synthetic sample",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Reasoning temperature; 0 is greedy"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic frames, sampler and action noise",
    )
    parser.add_argument("--cuda-graphs", action="store_true", help="Replay decode and expert_step")
    parser.add_argument("--runs", type=int, default=1, help="Repeat to measure warm timings")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with the PyTorch model (greedy only; loads the weights)",
    )
    args = parser.parse_args()
    if args.check and args.temperature != 0:
        parser.error("--check compares greedy reasoning; use --temperature 0")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.temperature >= 0:  # also rejects NaN
        parser.error("--temperature must be a number of at least 0")
    if args.temperature > torch.finfo(torch.float32).max:
        parser.error("--temperature exceeds the largest float32 value")

    host = HostPipeline(args.model_id)
    if args.manifest:
        from alpamayo2_edge.inference import load_clip_manifest
        from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

        clip = load_clip_manifest(args.manifest)
        sample = load_physical_aiavdataset(
            clip["clip_id"], t0_us=clip["t0_us"], nav_text=clip["nav_text"]
        )
    else:
        sample = synthetic_sample(args.seed)
    prompt = host.prepare(sample)
    runner = Runner(args.pte_dir, args.cuda_graphs)
    # Drawn on the GPU like the PyTorch model's diffusion noise, so --check can match it.
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(1, NUM_ACTION_TOKENS, 2, device="cuda", generator=noise_generator).cpu()
    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    for _ in range(args.runs):
        result = runner.run(host, prompt, args.temperature, noise, generator)
        rows.append(runner.timings)
        print(" ".join(f"{k} {v * 1000:.1f} ms" for k, v in runner.timings.items()))
    if args.runs > 1:
        warm = rows[1:]
        median = {k: statistics.median(r[k] for r in warm) for k in warm[0]}
        print(
            "median of warm runs: " + " ".join(f"{k} {v * 1000:.1f} ms" for k, v in median.items())
        )

    xyz, _ = host.trajectory(result["action"], prompt)
    for key, text in host.reasoning_text(result["generated"]).items():
        print(f"{key}: {text}")
    end = xyz[0, -1].tolist()
    print(f"trajectory end after 6.4 s: x {end[0]:.2f} m, y {end[1]:.2f} m")

    if args.check:
        # Frees the programs' GPU memory before the PyTorch model loads.
        del runner
        tokens, ref_xyz = upstream_reference(args.model_id, sample, args.seed)
        path_error = (xyz[0] - ref_xyz).norm(dim=-1).max().item()
        same = result["generated"] == tokens
        print(f"check: reasoning tokens {'equal' if same else 'DIFFER'}, ", end="")
        print(f"path max error {path_error * 100:.2f} cm")
        if not (same and xyz.isfinite().all() and path_error < MAX_PATH_ERROR_M):
            sys.exit(1)


if __name__ == "__main__":
    main()
