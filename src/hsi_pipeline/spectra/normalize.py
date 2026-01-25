"""Normalization utilities for spectral signatures."""

import numpy as np
from typing import Literal


NormMode = Literal["none", "minmax", "l2"]


def normalize_signature(
    values: np.ndarray,
    mode: NormMode = "none"
) -> np.ndarray:
    """Normalize spectral signature.
    
    Args:
        values: 1D array of spectral values.
        mode: Normalization mode:
            - 'none': No normalization
            - 'minmax': Scale to [0, 1]
            - 'l2': L2 normalization (unit length)
    
    Returns:
        Normalized values.
    
    Raises:
        ValueError: If mode is not supported.
    """
    if mode == "none":
        return values.copy()
    
    if mode == "minmax":
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    if mode == "l2":
        norm = np.linalg.norm(values)
        if norm < 1e-10:
            return np.zeros_like(values)
        return values / norm
    
    raise ValueError(f"Unknown normalization mode: {mode}. Use 'none', 'minmax', or 'l2'.")


def validate_normalize_mode(mode: str) -> NormMode:
    """Validate and return normalization mode.
    
    Args:
        mode: User-provided normalization mode string.
    
    Returns:
        Validated NormMode.
    
    Raises:
        ValueError: If mode is not supported.
    """
    if mode not in ("none", "minmax", "l2"):
        raise ValueError(
            f"Invalid normalization mode: '{mode}'. Use 'none', 'minmax', or 'l2'."
        )
    return mode
