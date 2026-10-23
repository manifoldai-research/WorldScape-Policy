"""Deterministic VLM image preprocessing shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def validate_vlm_preprocessing(value: object) -> dict[str, Any] | None:
    """Validate optional checkpoint metadata without changing legacy defaults."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("vlm_preprocessing must be a mapping or null")
    options = dict(value)
    unknown = set(options) - {"mode", "height", "width"}
    if unknown:
        raise ValueError(f"Unknown vlm_preprocessing fields: {sorted(unknown)}")
    mode = options.get("mode")
    if mode == "raw":
        if set(options) != {"mode"}:
            raise ValueError("Raw VLM preprocessing does not accept resize dimensions")
        return {"mode": "raw"}
    if mode != "resize":
        raise ValueError("vlm_preprocessing.mode must be 'raw' or 'resize'")
    result: dict[str, Any] = {"mode": "resize"}
    for key, default in (("height", 160), ("width", 320)):
        size = options.get(key, default)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"vlm_preprocessing.{key} must be a positive integer")
        result[key] = size
    return result


def preprocess_vlm_video(video: Tensor, config: object) -> Tensor:
    """Resize complete RGB frames, preserving leading dimensions and pixel range.

    uint8 training images and float [0,1] evaluation images use the same uint8
    interpolation/rounding so the VLM receives identical pixels on both paths.
    A null config preserves the caller's legacy preprocessing contract.
    """

    options = validate_vlm_preprocessing(config)
    if options is None or options["mode"] == "raw":
        return video
    if video.ndim < 4 or video.shape[-3] != 3:
        raise ValueError("VLM video must have shape [...,T,3,H,W]")
    if video.dtype != torch.uint8 and not video.is_floating_point():
        raise TypeError("VLM video must be uint8 or floating point in [0,1]")
    floating = video.is_floating_point()
    if floating and video.numel() and (
        not bool(torch.isfinite(video).all())
        or bool(video.amin() < 0)
        or bool(video.amax() > 1)
    ):
        raise ValueError("Floating-point VLM video must be finite and in [0,1]")
    pixels = (video.float() * 255).round().to(torch.uint8) if floating else video
    height, width = options["height"], options["width"]
    flat = pixels.reshape(-1, *pixels.shape[-3:])
    resized = TF.resize(
        flat, [height, width], interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).reshape(*video.shape[:-2], height, width)
    return resized.to(dtype=video.dtype).div(255) if floating else resized
