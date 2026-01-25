"""Spectral signature extraction functions."""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass


class ExtractionError(Exception):
    """Error during spectral extraction."""
    pass


class CoordinateError(ExtractionError):
    """Coordinates out of range."""
    pass


class ROIError(ExtractionError):
    """Invalid ROI for extraction."""
    pass


@dataclass
class SpectralSignature:
    """Extracted spectral signature with metadata."""
    values: np.ndarray  # 1D array of band values
    num_bands: int
    source: str  # 'pixel' or 'roi'
    artifact: str  # 'raw' or 'clean'
    
    # For pixel extraction
    pixel_x: Optional[int] = None
    pixel_y: Optional[int] = None
    
    # For ROI extraction
    roi_aggregation: Optional[str] = None
    roi_pixel_count: Optional[int] = None


def extract_pixel(
    hsi: np.ndarray,
    x: int,
    y: int,
    artifact_type: str = "raw"
) -> SpectralSignature:
    """Extract spectral signature from a single pixel.
    
    Args:
        hsi: HSI cube of shape (H, W, C).
        x: X coordinate (column index).
        y: Y coordinate (row index).
        artifact_type: Source artifact type.
    
    Returns:
        SpectralSignature with values for all bands.
    
    Raises:
        CoordinateError: If coordinates are out of range.
    """
    if hsi.ndim != 3:
        raise ExtractionError(f"HSI must be 3D (H, W, C), got {hsi.ndim}D")
    
    h, w, c = hsi.shape
    
    if x < 0 or x >= w:
        raise CoordinateError(
            f"X coordinate {x} out of range. Valid range: 0 to {w-1}"
        )
    
    if y < 0 or y >= h:
        raise CoordinateError(
            f"Y coordinate {y} out of range. Valid range: 0 to {h-1}"
        )
    
    values = hsi[y, x, :]  # Note: y=row, x=col
    
    return SpectralSignature(
        values=values.copy(),
        num_bands=c,
        source="pixel",
        artifact=artifact_type,
        pixel_x=x,
        pixel_y=y
    )


def extract_roi_aggregate(
    hsi: np.ndarray,
    mask: np.ndarray,
    aggregation: Literal["mean", "median"] = "mean",
    artifact_type: str = "raw"
) -> SpectralSignature:
    """Extract aggregated spectral signature from ROI.
    
    Args:
        hsi: HSI cube of shape (H, W, C).
        mask: Binary mask of shape (H, W). True = ROI pixels.
        aggregation: Aggregation method ('mean' or 'median').
        artifact_type: Source artifact type.
    
    Returns:
        SpectralSignature with aggregated values.
    
    Raises:
        ROIError: If ROI is invalid, empty, or dimensions don't match.
    """
    if hsi.ndim != 3:
        raise ExtractionError(f"HSI must be 3D (H, W, C), got {hsi.ndim}D")
    
    if mask.shape != hsi.shape[:2]:
        raise ROIError(
            f"Mask shape {mask.shape} doesn't match HSI spatial dims {hsi.shape[:2]}"
        )
    
    roi_count = np.sum(mask)
    
    if roi_count == 0:
        raise ROIError("ROI mask is empty (no True pixels)")
    
    if aggregation not in ("mean", "median"):
        raise ValueError(
            f"Unknown aggregation: {aggregation}. Use 'mean' or 'median'."
        )
    
    roi_pixels = hsi[mask]  # Shape: (N, C)
    
    if aggregation == "mean":
        values = np.mean(roi_pixels, axis=0)
    else:
        values = np.median(roi_pixels, axis=0)
    
    return SpectralSignature(
        values=values.astype(np.float32),
        num_bands=hsi.shape[2],
        source="roi",
        artifact=artifact_type,
        roi_aggregation=aggregation,
        roi_pixel_count=int(roi_count)
    )
