import torch

from alpamayo2_edge.diffusion.flow_matching import FlowMatching


def test_seeded_diffusion_is_independent_of_global_rng():
    diffusion = FlowMatching(x_dims=(4, 2))

    def draw():
        return diffusion.sample(
            1,
            lambda x, t: torch.ones_like(x),
            inference_step=5,
            generator=torch.Generator().manual_seed(43),
        )

    first = draw()
    torch.randn(100)
    second = draw()
    assert torch.equal(first, second)
    noise = torch.randn((1, 4, 2), generator=torch.Generator().manual_seed(43))
    torch.testing.assert_close(first, noise + 1)
