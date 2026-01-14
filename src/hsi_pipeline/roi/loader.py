"""ROI mask loader and validator."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


BINARIZE_THRESHOLD = 127


@dataclass
class ROIMaskResult:
    """Result of loading and validating ROI mask."""
    mask: np.ndarray  # Binary mask (H, W), dtype bool
    coverage: float   # Fraction of pixels that are ROI (0-1)
    warnings: list[str]
    path: str


class ROILoadError(Exception):
    """Raised when ROI mask cannot be loaded."""
    pass


class ROIValidationError(Exception):
    """Raised when ROI mask fails validation."""
    pass


def load_roi_mask(
    mask_path: Path,
    expected_shape: tuple[int, int]
) -> ROIMaskResult:
    """Load and validate ROI mask.
    
    Args:
        mask_path: Path to mask image file.
        expected_shape: Expected (H, W) shape matching the input image.
    
    Returns:
        ROIMaskResult with binary mask and metadata.
    
    Raises:
        ROILoadError: If mask cannot be loaded.
        ROIValidationError: If mask dimensions don't match.
    """
    if not mask_path.exists():
        raise ROILoadError(f"ROI mask not found: {mask_path}")
    
    try:
        with Image.open(mask_path) as img:
            mask = np.array(img.convert("L"))  # Convert to grayscale
    except Exception as e:
        raise ROILoadError(f"Cannot decode ROI mask: {e}")
    
    warnings = []
    
    # Validate dimensions
    if mask.shape != expected_shape:
        raise ROIValidationError(
            f"ROI mask size mismatch: expected {expected_shape}, got {mask.shape}"
        )
    
    # Check if binary
    unique_values = np.unique(mask)
    is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0, 255})
    
    if not is_binary:
        warnings.append(
            f"Non-binary mask detected ({len(unique_values)} values), "
            f"binarizing at threshold {BINARIZE_THRESHOLD}"
        )
        mask = (mask > BINARIZE_THRESHOLD).astype(np.uint8) * 255
    
    # Convert to boolean
    binary_mask = mask > 0
    
    # Calculate coverage
    coverage = float(np.mean(binary_mask))
    
    # Warn on edge cases
    if coverage == 0.0:
        warnings.append("ROI mask is empty (0% coverage), separability will be NA")
    elif coverage == 1.0:
        warnings.append("ROI mask is full (100% coverage), separability will be NA")
    
    return ROIMaskResult(
        mask=binary_mask,
        coverage=coverage,
        warnings=warnings,
        path=str(mask_path)
    )
