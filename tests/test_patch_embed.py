"""The GEMM patch embedding against the checkpoint's Conv3d."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

from alpamayo2_edge.accel import use_linear_patch_embedding


def _model_with(embed):
    return SimpleNamespace(
        vlm=SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(patch_embed=embed)))
    )


def test_linear_patch_embedding_matches_conv3d():
    torch.manual_seed(1)
    embed = Qwen3VLVisionPatchEmbed(
        Qwen3VLVisionConfig(hidden_size=32, patch_size=2, temporal_patch_size=2)
    )
    patches = torch.randn(40, 3 * 2 * 2 * 2)
    with torch.no_grad():
        expected = embed(patches)
        use_linear_patch_embedding(_model_with(embed))
        actual = embed(patches)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    del embed.forward
    torch.testing.assert_close(embed(patches), expected, rtol=0, atol=0)


def test_linear_patch_embedding_matches_conv3d_in_bfloat16():
    from alpamayo2_edge.accel.patch_embed import _linear_patch_embed_forward

    torch.manual_seed(2)
    embed = Qwen3VLVisionPatchEmbed(Qwen3VLVisionConfig(hidden_size=64)).to(torch.bfloat16)
    patches = torch.randn(64, 3 * 2 * 16 * 16)
    with torch.no_grad():
        expected = embed(patches).float()
        actual = _linear_patch_embed_forward(embed, patches).float()
        exact = torch.nn.functional.linear(
            patches.to(torch.bfloat16).double(),
            embed.proj.weight.double().reshape(64, -1),
            embed.proj.bias.double(),
        )
    # Both round one FP32-accumulated dot product per output to BF16.
    assert (actual - expected).abs().max() <= 2 * (exact.abs().max() * 2**-8)
    torch.testing.assert_close(actual, exact.float(), rtol=2**-7, atol=1e-2)


def test_linear_patch_embedding_requires_kernel_equal_to_stride():
    embed = SimpleNamespace(proj=torch.nn.Conv3d(3, 8, kernel_size=(2, 2, 2), stride=(1, 2, 2)))
    with pytest.raises(ValueError, match="kernel == stride"):
        use_linear_patch_embedding(_model_with(embed))
