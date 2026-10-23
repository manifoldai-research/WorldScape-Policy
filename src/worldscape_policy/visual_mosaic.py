from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from worldscape_policy.memory.visual.normalization import VisualInputRange


DiffusionViewLayout = Literal["mosaic_2x2", "mosaic_T"]


def validate_diffusion_view_layout(
    value: object, *, path: str = "diffusion_view_layout"
) -> DiffusionViewLayout:
    """Validate a supported diffusion layout without rewriting its name."""

    if not isinstance(value, str):
        raise ValueError(f"{path} must be 'mosaic_2x2' or 'mosaic_T'")
    if value not in {"mosaic_2x2", "mosaic_T"}:
        raise ValueError(f"{path} must be 'mosaic_2x2' or 'mosaic_T'")
    return cast(DiffusionViewLayout, value)


def compose_three_view_mosaic(views: Tensor) -> Tensor:
    """Place ``[head, left, right]`` views in the canonical 2x2 layout."""

    if views.ndim != 6:
        raise ValueError("views must have shape [B,T,V,C,H,W]")
    if views.shape[2] != 3:
        raise ValueError("diffusion video requires exactly [head, left, right]")
    batch, frames, _, channels, height, width = views.shape
    mosaic = views.new_zeros((batch, frames, channels, 2 * height, 2 * width))
    mosaic[..., :height, :width] = views[:, :, 0]
    mosaic[..., height:, :width] = views[:, :, 1]
    mosaic[..., :height, width:] = views[:, :, 2]
    return mosaic


def prepare_diffusion_mosaic(
    views: Tensor,
    *,
    input_range: VisualInputRange,
) -> Tensor:
    """Compose, normalize, then resize the full mosaic to one-view resolution."""

    from worldscape_policy.memory.visual.normalization import normalize_visual

    height, width = views.shape[-2:]
    mosaic = normalize_visual(
        compose_three_view_mosaic(views),
        input_range=input_range,
    )
    batch, frames, channels, mosaic_height, mosaic_width = mosaic.shape
    resized = F.interpolate(
        mosaic.reshape(batch * frames, channels, mosaic_height, mosaic_width),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, height, width)


def compose_three_view_mosaic_T(views: Tensor) -> Tensor:
    """Resize raw ``[head, left, right]`` views directly into a 384x320 image."""

    if views.ndim != 6:
        raise ValueError("views must have shape [B,T,V,C,H,W]")
    if views.shape[2] != 3:
        raise ValueError("diffusion video requires exactly [head, left, right]")
    batch, frames, _, channels, height, width = views.shape
    flat = views.reshape(batch * frames, 3, channels, height, width)

    def resize(view: Tensor, size: tuple[int, int]) -> Tensor:
        if not view.is_floating_point():
            view = view.float()
        return F.interpolate(
            view,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    head = resize(flat[:, 0], (256, 320))
    left = resize(flat[:, 1], (128, 160))
    right = resize(flat[:, 2], (128, 160))
    bottom = torch.cat((left, right), dim=-1)
    result = torch.cat((head, bottom), dim=-2)
    return result.reshape(batch, frames, channels, 384, 320)


def prepare_diffusion_mosaic_T(
    views: Tensor,
    *,
    input_range: VisualInputRange,
) -> Tensor:
    """Normalize raw views, then resize each view once into the final layout."""

    from worldscape_policy.memory.visual.normalization import normalize_visual

    normalized = normalize_visual(views, input_range=input_range)
    # Antialiased interpolation may overshoot the normalized range by a few
    # ulps around saturated edges, while the VAE input contract is strict.
    return compose_three_view_mosaic_T(normalized).clamp_(-1.0, 1.0)


__all__ = [
    "DiffusionViewLayout",
    "validate_diffusion_view_layout",
    "compose_three_view_mosaic",
    "compose_three_view_mosaic_T",
    "prepare_diffusion_mosaic",
    "prepare_diffusion_mosaic_T",
]
