# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export-friendly Alpamayo 2 Edge modules for Torch-TensorRT and ExecuTorch.

They reuse the weights and submodules of the loaded model and only change how the
forward passes are written:

* The language model keeps its KV cache in buffers of shape
  ``[1, kv_heads, max_seq_len, head_dim]``. ``prefill``, ``decode`` and ``expert_step``
  write the same buffers with ``index_copy_``, and each TensorRT engine updates them in
  place.
* ``decode`` and ``expert_step`` attend over the whole cache with a mask, so they have
  fixed shapes and can replay as CUDA graphs.
* ``prefill`` and ``decode`` return the next token id, sampled inside the engine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.cosmos3_edge.modeling_cosmos3_edge import apply_rotary_pos_emb

NUM_ACTION_TOKENS = 64


def sample(logits: torch.Tensor, temperature: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Temperature sampling with the Gumbel-max trick; a temperature of 0 is greedy.

    The noise is an input, ``uniform`` in [0, 1) with the shape of ``logits``, because
    TensorRT cannot generate random numbers. Returns ``[1, 1]`` int64.
    """
    # Centered first, so a tiny temperature cannot overflow large logits.
    centered = logits - logits.amax(dim=-1, keepdim=True)
    scaled = centered / torch.where(temperature > 0, temperature, 1.0)
    gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
    sampled = (scaled + gumbel).argmax(dim=-1, keepdim=True)
    greedy = logits.argmax(dim=-1, keepdim=True)
    return torch.where(temperature > 0, sampled, greedy).to(torch.int64)


def _layer_norm_fp32(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """A LayerNorm computed and returned in FP32, as under autocast."""
    return F.layer_norm(
        x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps
    )


class KVCache(nn.Module):
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(shape, dtype=dtype))
        self.register_buffer("v", torch.zeros(shape, dtype=dtype))


class StaticCacheModel(nn.Module):
    """Language model and action expert over one static KV cache."""

    def __init__(self, model: nn.Module, max_seq_len: int, blocked_token_ids: list[int]) -> None:
        super().__init__()
        self.text_model = model.vlm.model.language_model
        self.lm_head = model.vlm.lm_head
        self.action_expert = model.expert
        self.image_token_id = model.vlm.config.image_token_id
        config = self.text_model.config
        self.head_dim = config.head_dim
        if model.expert.action_space.get_action_space_dims()[0] != NUM_ACTION_TOKENS:
            raise ValueError(f"The expert must predict {NUM_ACTION_TOKENS} waypoints")
        dtype = self.lm_head.weight.dtype
        shape = (1, config.num_key_value_heads, max_seq_len, self.head_dim)
        self.caches = nn.ModuleList(KVCache(shape, dtype) for _ in range(config.num_hidden_layers))
        self.register_buffer("cache_positions", torch.arange(max_seq_len), persistent=False)
        blocked = torch.zeros(self.lm_head.weight.shape[0], dtype=torch.bool)
        blocked[blocked_token_ids] = True
        self.register_buffer("blocked_tokens", blocked, persistent=False)

    def text(
        self,
        embeds: torch.Tensor,
        pos: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: torch.Tensor,
        uniform: torch.Tensor,
        prompt: bool,
    ) -> torch.Tensor:
        """Write T tokens into the cache at ``input_pos`` and sample the next token.

        embeds [1, T, hidden], pos [3, 1, T] mRoPE positions, input_pos [1]. With
        ``prompt``, the tokens are the whole prompt from position 0 and attention is
        causal over them; otherwise it covers the valid cache columns.
        """
        tm = self.text_model
        seq_len = embeds.shape[1]
        cos, sin = tm.rotary_emb(embeds, pos)
        idx = input_pos + torch.arange(seq_len, device=embeds.device)
        if not prompt:
            mask = (self.cache_positions[None, :] <= idx[:, None])[None, None]
        h = embeds
        for layer, cache in zip(tm.layers, self.caches):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(1, seq_len, -1, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(1, seq_len, -1, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(1, seq_len, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cache.k.index_copy_(2, idx, k)
            cache.v.index_copy_(2, idx, v)
            if prompt:
                # Reads the keys back from the cache: TensorRT updates a cache in place
                # only when the graph reads the written values.
                o = F.scaled_dot_product_attention(
                    q,
                    cache.k[:, :, :seq_len],
                    cache.v[:, :, :seq_len],
                    is_causal=True,
                    scale=attn.scaling,
                    enable_gqa=True,
                )
            else:
                o = F.scaled_dot_product_attention(
                    q, cache.k, cache.v, attn_mask=mask, scale=attn.scaling, enable_gqa=True
                )
            h = h + attn.o_proj(o.transpose(1, 2).reshape(1, seq_len, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.lm_head(tm.norm(h[:, -1])).float()
        logits = logits.masked_fill(self.blocked_tokens, float("-inf"))
        return sample(logits, temperature, uniform)

    def action_in(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The action input projection, its layer norm in FP32 as under autocast."""
        proj = self.action_expert.action_in_proj
        n = x.shape[1]
        feats = torch.cat([enc(x[:, :, j]) for j, enc in enumerate(proj.sinus)], dim=-1)
        time_feats = proj.timestep_fourier_encoder(t[..., -1]).repeat(1, n, 1)
        h = torch.cat((feats, time_feats), dim=-1)
        h = proj.encoder(h.to(proj.encoder.trunk[0].weight.dtype).flatten(0, 1))
        return _layer_norm_fp32(proj.norm, h.reshape(1, n, -1))


class Prefill(nn.Module):
    """The whole prompt in one call; image rows replace the image placeholder tokens."""

    def __init__(self, core: StaticCacheModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, tokens, image_embeds, pos, temperature, uniform):
        embeds = self.core.text_model.embed_tokens(tokens)
        is_image = (tokens == self.core.image_token_id).unsqueeze(-1)
        embeds = torch.where(is_image, image_embeds, embeds)
        start = torch.zeros(1, dtype=torch.int64, device=tokens.device)
        return self.core.text(embeds, pos, start, temperature, uniform, prompt=True)


class Decode(nn.Module):
    """One reasoning token. Text positions advance by one on all mRoPE axes."""

    def __init__(self, core: StaticCacheModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, token, input_pos, rope_delta, temperature, uniform):
        embeds = self.core.text_model.embed_tokens(token)
        pos = (input_pos + rope_delta).view(1, 1, 1).expand(3, 1, 1)
        return self.core.text(embeds, pos, input_pos, temperature, uniform, prompt=False)


class ExpertStep(nn.Module):
    """One Euler step of the flow-matching expert: x [1, 64, 2] fp32, t and dt [1] fp32.

    A TensorRT engine updates a cache in place once per call, and every step writes
    the action tokens' keys and values, so the caller runs this once per step.
    """

    def __init__(self, core: StaticCacheModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, x, t, dt, prefix_len, rope_delta):
        core = self.core
        ex = core.action_expert
        n = x.shape[1]
        steps = torch.arange(n, device=x.device)
        pos = (steps + prefix_len + rope_delta).view(1, 1, n).expand(3, 1, n)
        cos, sin = ex.expert.rotary_emb(x, pos)
        # Like upstream's accelerated path, the action tokens write columns
        # [prefix_len, prefix_len + 64) and attend to all valid columns without a causal mask.
        idx = prefix_len + steps
        mask = (core.cache_positions < prefix_len + n).expand(1, 1, n, -1)
        dtype = core.caches[0].k.dtype
        # FP32 residual stream, matmuls and attention in the weight dtype, as upstream
        # under BF16 autocast.
        h = core.action_in(x, t.view(1, 1, 1))
        for layer, cache in zip(ex.expert.layers, core.caches):
            attn = layer.self_attn
            xb = layer.input_layernorm(h).to(dtype)
            q = attn.q_proj(xb).view(1, n, -1, core.head_dim)
            k = attn.k_proj(xb).view(1, n, -1, core.head_dim)
            v = attn.v_proj(xb).view(1, n, -1, core.head_dim)
            q, k = apply_rotary_pos_emb(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(), cos, sin
            )
            cache.k.index_copy_(2, idx, k.to(dtype))
            cache.v.index_copy_(2, idx, v.transpose(1, 2))
            o = F.scaled_dot_product_attention(
                q.to(dtype), cache.k, cache.v, attn_mask=mask, scale=attn.scaling, enable_gqa=True
            )
            h = h + attn.o_proj(o.transpose(1, 2).reshape(1, n, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h).to(dtype))
        velocity = ex.action_out_proj(ex.expert.norm(h).to(dtype))
        return x + dt * velocity.float()


class VisionEncoder(nn.Module):
    """Vision tower and projector for a batch of images that share one patch grid."""

    def __init__(self, model: nn.Module, grid_h: int, grid_w: int) -> None:
        super().__init__()
        vm = model.vlm.model
        self.visual, self.projector = vm.visual, vm.projector
        emb = self.visual.embeddings
        size = emb.position_embedding_size
        with torch.no_grad():
            table = emb.position_embedding.weight.reshape(size, size, -1)
            # Resized in FP32 like upstream under autocast.
            pos_embed = emb.resize_positional_embeddings(
                table.float(), torch.tensor([[1, grid_h, grid_w]]), emb.config.spatial_merge_size
            )
        self.register_buffer("pos_embed", pos_embed.to(table.dtype), persistent=False)
        self.merge = emb.config.spatial_merge_size**2

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches [B, H*W, 768] normalized fp32 -> [B, H*W/4, hidden]."""
        batch, num_patches, _ = patches.shape
        v = self.visual
        dtype = v.embeddings.patch_embedding.weight.dtype
        h = v.embeddings.patch_embedding(patches.to(dtype)) + self.pos_embed
        for layer in v.encoder.layers:
            attn = layer.self_attn
            # TensorRT computes layer norms and GELU in their input type, where PyTorch uses
            # FP32 inside. In BF16 that moved the predicted path several times more.
            x = _layer_norm_fp32(layer.layer_norm1, h).to(dtype)
            shape = (batch, num_patches, attn.num_heads, attn.head_dim)
            q = attn.q_proj(x).view(shape).transpose(1, 2)
            k = attn.k_proj(x).view(shape).transpose(1, 2)
            vv = attn.v_proj(x).view(shape).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, vv, scale=attn.scale)
            h = h + attn.out_proj(o.transpose(1, 2).reshape(batch, num_patches, -1))
            mlp = layer.mlp
            y = mlp.fc1(_layer_norm_fp32(layer.layer_norm2, h).to(dtype))
            h = h + mlp.fc2(mlp.activation_fn(y.float()).to(dtype))
        p = self.projector
        h = _layer_norm_fp32(p.norm, _layer_norm_fp32(v.post_layernorm, h))
        h = h.to(dtype).reshape(-1, p.hidden_size)
        out = p.linear_fc2(p.act_fn(p.linear_fc1(h)))
        return out.reshape(batch, num_patches // self.merge, -1)


def build_methods(
    model: nn.Module,
    max_seq_len: int,
    blocked_token_ids: list[int],
    vision_grids: dict[str, tuple[int, int]],
) -> dict[str, nn.Module]:
    """All exported methods. The text methods share one ``StaticCacheModel``."""
    core = StaticCacheModel(model, max_seq_len, blocked_token_ids)
    methods: dict[str, nn.Module] = {
        name: VisionEncoder(model, gh, gw) for name, (gh, gw) in vision_grids.items()
    }
    methods.update(prefill=Prefill(core), decode=Decode(core), expert_step=ExpertStep(core))
    return {name: m.eval() for name, m in methods.items()}
