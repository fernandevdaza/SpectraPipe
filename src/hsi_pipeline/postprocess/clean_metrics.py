"""Metrics for clean HSI evaluation."""

import numpy as np
from typing import Optional, Dict, Any


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Compute spectral angle between two spectra in radians."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    cos_angle = np.dot(a, b) / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)


def calculate_clean_metrics(
    hsi_raw: np.ndarray,
    hsi_clean: np.ndarray,
    mask: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Calculate metrics comparing raw and clean HSI.
    
    Args:
        hsi_raw: Original HSI cube (H, W, C).
        hsi_clean: Cleaned HSI cube (H, W, C).
        mask: Binary ROI mask (H, W). True = ROI.
    
    Returns:
        Dict with clean_separability, raw_clean_sam, raw_clean_rmse.
        Returns None if metrics cannot be computed.
    """
    if hsi_raw.shape != hsi_clean.shape:
        raise ValueError("HSI shapes must match")
    
    if mask.shape != hsi_raw.shape[:2]:
        raise ValueError("Mask shape must match HSI spatial dims")
    
    roi_count = np.sum(mask)
    bg_count = np.sum(~mask)
    
    if roi_count == 0 or bg_count == 0:
        return None
    
    roi_raw = hsi_raw[mask].astype(np.float32)
    roi_clean = hsi_clean[mask].astype(np.float32)
    bg_clean = hsi_clean[~mask].astype(np.float32)
    
    roi_mean_clean = np.mean(roi_clean, axis=0)
    bg_mean_clean = np.mean(bg_clean, axis=0)
    
    roi_mean_raw = np.mean(roi_raw, axis=0)
    
    sam_value = spectral_angle(roi_mean_raw, roi_mean_clean)
    
    rmse_value = np.sqrt(np.mean((roi_raw - roi_clean) ** 2))
    
    clean_sep = spectral_angle(roi_mean_clean, bg_mean_clean)
    
    return {
        "clean_separability": float(clean_sep),
        "raw_clean_sam": float(sam_value),
        "raw_clean_rmse": float(rmse_value),
    }
