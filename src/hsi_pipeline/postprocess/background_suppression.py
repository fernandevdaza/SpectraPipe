"""Background suppression for HSI clean generation."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CleanResult:
    """Result of background suppression."""
    hsi_clean: np.ndarray
    policy: str
    bg_mean_spectrum: np.ndarray


def suppress_background(
    hsi: np.ndarray,
    mask: np.ndarray,
    policy: str = "subtract_mean"
) -> Optional[CleanResult]:
    """Apply background suppression to HSI using ROI mask.
    
    Args:
        hsi: Input HSI cube of shape (H, W, C).
        mask: Binary ROI mask of shape (H, W). True = ROI, False = background.
        policy: Suppression policy. Options:
            - "subtract_mean": Subtract background mean spectrum from all pixels
            - "zero_background": Set background pixels to zero
    
    Returns:
        CleanResult with cleaned HSI, or None if cannot compute (empty/full ROI).
    
    Raises:
        ValueError: If shapes don't match or invalid policy.
    """
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3D (H, W, C), got {hsi.ndim}D")
    
    if mask.shape != hsi.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} doesn't match HSI spatial dims {hsi.shape[:2]}"
        )
    
    roi_count = np.sum(mask)
    bg_count = np.sum(~mask)
    
    if roi_count == 0 or bg_count == 0:
        return None
    
    bg_pixels = hsi[~mask]
    bg_mean = np.mean(bg_pixels, axis=0)
    
    if policy == "subtract_mean":
        hsi_clean = hsi.astype(np.float32) - bg_mean
        hsi_clean = np.clip(hsi_clean, 0, None)
        hsi_clean = hsi_clean.astype(hsi.dtype)
    elif policy == "zero_background":
        hsi_clean = hsi.copy()
        hsi_clean[~mask] = 0
    else:
        raise ValueError(f"Unknown policy: {policy}. Use 'subtract_mean' or 'zero_background'")
    
    return CleanResult(
        hsi_clean=hsi_clean,
        policy=policy,
        bg_mean_spectrum=bg_mean
    )
