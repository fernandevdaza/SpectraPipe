"""Separability calculation for ROI vs background."""

import numpy as np


def calculate_separability(
    hsi: np.ndarray,
    mask: np.ndarray
) -> float | None:
    """Calculate spectral separability between ROI and background.
    
    Uses Jeffries-Matusita distance approximation based on mean spectra
    and spectral angle.
    
    Args:
        hsi: HSI cube of shape (H, W, C).
        mask: Binary mask of shape (H, W), True = ROI.
    
    Returns:
        Separability score (0-1) or None if ROI is empty/full.
    """
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3D (H, W, C), got {hsi.ndim}D")
    
    if mask.shape != hsi.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} doesn't match HSI spatial dims {hsi.shape[:2]}"
        )
    
    # Check for empty/full ROI
    roi_pixels = np.sum(mask)
    total_pixels = mask.size
    bg_pixels = total_pixels - roi_pixels
    
    if roi_pixels == 0 or bg_pixels == 0:
        return None  # Cannot compute separability
    
    # Flatten spatial dimensions
    h, w, c = hsi.shape
    flat_hsi = hsi.reshape(-1, c)  # (H*W, C)
    flat_mask = mask.ravel()  # (H*W,)
    
    # Extract ROI and background spectra
    roi_spectra = flat_hsi[flat_mask]  # (N_roi, C)
    bg_spectra = flat_hsi[~flat_mask]  # (N_bg, C)
    
    # Compute mean spectra
    roi_mean = np.mean(roi_spectra, axis=0)  # (C,)
    bg_mean = np.mean(bg_spectra, axis=0)    # (C,)
    
    # Compute Spectral Angle Mapper (SAM) between means
    dot_product = np.dot(roi_mean, bg_mean)
    norm_roi = np.linalg.norm(roi_mean)
    norm_bg = np.linalg.norm(bg_mean)
    
    if norm_roi == 0 or norm_bg == 0:
        return None  # Degenerate spectra
    
    cos_angle = dot_product / (norm_roi * norm_bg)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    angle_rad = np.arccos(cos_angle)
    
    # Normalize to 0-1 range (0 = identical, 1 = orthogonal)
    separability = angle_rad / (np.pi / 2)
    separability = np.clip(separability, 0.0, 1.0)
    
    return float(separability)
