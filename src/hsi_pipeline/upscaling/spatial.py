"""Spatial upscaling for HSI data."""

import numpy as np
from scipy.ndimage import zoom


def upscale_baseline(hsi: np.ndarray, factor: int = 2) -> np.ndarray:
    """Upscale HSI using bicubic interpolation (baseline method).
    
    Args:
        hsi: Input HSI cube of shape (H, W, C).
        factor: Upscaling factor (default 2x).
    
    Returns:
        Upscaled HSI of shape (H*factor, W*factor, C).
    
    Raises:
        ValueError: If input is not 3D or factor < 1.
    """
    if hsi.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got {hsi.ndim}D")
    
    if factor < 1:
        raise ValueError(f"Factor must be >= 1, got {factor}")
    
    if factor == 1:
        return hsi.copy()
    
    # Bicubic interpolation: zoom spatial dimensions only
    zoom_factors = (factor, factor, 1)  # (H, W, C)
    upscaled = zoom(hsi, zoom_factors, order=3)  # order=3 = bicubic
    
    return upscaled.astype(hsi.dtype)


def upscale_improved(
    hsi: np.ndarray,
    rgb_guide: np.ndarray,
    factor: int = 2
) -> np.ndarray:
    """Upscale HSI using guided upsampling with RGB reference.
    
    Uses bicubic for now, but structure allows future improvement
    with edge-aware or learned methods.
    
    Args:
        hsi: Input HSI cube of shape (H, W, C).
        rgb_guide: High-resolution RGB guide of shape (H*factor, W*factor, 3).
        factor: Upscaling factor.
    
    Returns:
        Upscaled HSI of shape (H*factor, W*factor, C).
    
    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3D (H, W, C), got {hsi.ndim}D")
    
    if rgb_guide.ndim != 3:
        raise ValueError(f"RGB guide must be 3D (H, W, 3), got {rgb_guide.ndim}D")
    
    expected_h = hsi.shape[0] * factor
    expected_w = hsi.shape[1] * factor
    
    if rgb_guide.shape[0] != expected_h or rgb_guide.shape[1] != expected_w:
        raise ValueError(
            f"RGB guide shape mismatch: expected ({expected_h}, {expected_w}, 3), "
            f"got {rgb_guide.shape}"
        )
    
    # For now, use baseline bicubic (future: edge-aware guided filter)
    upscaled = upscale_baseline(hsi, factor)
    
    return upscaled
