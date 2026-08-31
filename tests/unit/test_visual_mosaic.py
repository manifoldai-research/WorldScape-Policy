from __future__ import annotations

import torch

from worldscape_policy.visual_mosaic import (
    compose_three_view_mosaic,
    prepare_diffusion_mosaic,
)


def test_three_view_mosaic_matches_canonical_quadrant_layout() -> None:
    views = torch.stack(
        (
            torch.full((1, 1, 1, 2, 3), 10, dtype=torch.uint8),
            torch.full((1, 1, 1, 2, 3), 20, dtype=torch.uint8),
            torch.full((1, 1, 1, 2, 3), 30, dtype=torch.uint8),
        ),
        dim=2,
    )

    mosaic = compose_three_view_mosaic(views)

    assert mosaic.shape == (1, 1, 1, 4, 6)
    assert torch.equal(mosaic[..., :2, :3], torch.full((1, 1, 1, 2, 3), 10))
    assert torch.equal(mosaic[..., 2:, :3], torch.full((1, 1, 1, 2, 3), 20))
    assert torch.equal(mosaic[..., :2, 3:], torch.full((1, 1, 1, 2, 3), 30))
    assert torch.count_nonzero(mosaic[..., 2:, 3:]) == 0


def test_diffusion_mosaic_normalizes_and_resizes_to_single_view_shape() -> None:
    views = torch.zeros((2, 3, 3, 3, 4, 8), dtype=torch.uint8)
    views[:, :, 0] = 255

    result = prepare_diffusion_mosaic(views, input_range="uint8")

    assert result.shape == (2, 3, 3, 4, 8)
    assert result.dtype == torch.float32
    assert result.amin() >= -1
    assert result.amax() <= 1
