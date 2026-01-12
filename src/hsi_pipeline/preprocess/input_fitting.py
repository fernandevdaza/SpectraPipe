"""Input fitting utilities for MST++ compatibility."""

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class FittingResult:
    """Result of input fitting operation."""
    fitted: np.ndarray
    original_shape: tuple[int, int]
    fitted_shape: tuple[int, int]
    policy: str
    padding: tuple[int, int, int, int]  # top, bottom, left, right


def fit_input(
    rgb: np.ndarray,
    multiple: int = 32,
    policy: Literal["pad_to_multiple"] = "pad_to_multiple"
) -> FittingResult:
    """Fit input image dimensions to be compatible with MST++.
    
    MST++ requires input dimensions to be multiples of 32 due to
    encoder stride requirements. This function pads the image
    symmetrically using reflect padding.
    
    Args:
        rgb: Input RGB image (H, W, 3).
        multiple: Target multiple for dimensions.
        policy: Fitting policy name.
    
    Returns:
        FittingResult with fitted image and metadata.
    
    Raises:
        ValueError: If input has invalid dimensions.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got shape {rgb.shape}")
    
    h, w = rgb.shape[:2]
    
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid dimensions: {h}x{w}")
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    if pad_h == 0 and pad_w == 0:
        return FittingResult(
            fitted=rgb,
            original_shape=(h, w),
            fitted_shape=(h, w),
            policy=policy,
            padding=(0, 0, 0, 0)
        )
    
    fitted = np.pad(
        rgb,
        ((top, bottom), (left, right), (0, 0)),
        mode="reflect"
    )
    
    return FittingResult(
        fitted=fitted,
        original_shape=(h, w),
        fitted_shape=fitted.shape[:2],
        policy=policy,
        padding=(top, bottom, left, right)
    )


def unfit_output(
    hsi: np.ndarray,
    original_shape: tuple[int, int],
    padding: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop HSI output back to original dimensions.
    
    Args:
        hsi: HSI cube (H, W, C) after inference.
        original_shape: Original (H, W) before fitting.
        padding: (top, bottom, left, right) padding that was applied.
    
    Returns:
        Cropped HSI cube with original spatial dimensions.
    """
    top, bottom, left, right = padding
    
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return hsi
    
    h_end = hsi.shape[0] - bottom if bottom > 0 else hsi.shape[0]
    w_end = hsi.shape[1] - right if right > 0 else hsi.shape[1]
    
    return hsi[top:h_end, left:w_end, :]
