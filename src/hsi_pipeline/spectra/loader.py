"""HSI artifact loader for spectral extraction."""

import numpy as np
from pathlib import Path
from typing import Literal
from dataclasses import dataclass


class HSILoadError(Exception):
    """Error loading HSI artifact."""
    pass


class HSINotFoundError(HSILoadError):
    """HSI artifact not found."""
    pass


@dataclass
class LoadedHSI:
    """Loaded HSI data with metadata."""
    data: np.ndarray
    path: Path
    artifact_type: str  # 'raw' or 'clean'
    shape: tuple


ARTIFACT_FILENAMES = {
    "raw": "hsi_raw_full",
    "clean": "hsi_clean_full",
}


def load_hsi_artifact(
    from_dir: Path,
    artifact: Literal["raw", "clean"] = "raw"
) -> LoadedHSI:
    """Load an HSI artifact from a pipeline output directory.
    
    Args:
        from_dir: Directory containing pipeline outputs.
        artifact: Which artifact to load ('raw' or 'clean').
    
    Returns:
        LoadedHSI with data and metadata.
    
    Raises:
        HSINotFoundError: If artifact file doesn't exist.
        HSILoadError: If artifact can't be loaded.
    """
    from_dir = Path(from_dir)
    
    if not from_dir.exists():
        raise HSINotFoundError(f"Output directory not found: {from_dir}")
    
    if not from_dir.is_dir():
        raise HSINotFoundError(f"Path is not a directory: {from_dir}")
    
    if artifact not in ARTIFACT_FILENAMES:
        raise ValueError(f"Unknown artifact type: {artifact}. Use 'raw' or 'clean'.")
    
    base_name = ARTIFACT_FILENAMES[artifact]
    
    # Try NPZ first, then NPY
    npz_path = from_dir / f"{base_name}.npz"
    npy_path = from_dir / f"{base_name}.npy"
    
    if npz_path.exists():
        try:
            loaded = np.load(npz_path, allow_pickle=True)
            
            # NPZ Schema v1: use 'cube' key
            if "cube" in loaded:
                data = loaded["cube"]
            # Legacy: use 'data' key
            elif "data" in loaded:
                import warnings
                warnings.warn(
                    f"Legacy NPZ detected: '{npz_path.name}' uses 'data' instead of 'cube'. "
                    "Re-export to update to schema v1.",
                    DeprecationWarning
                )
                data = loaded["data"]
            else:
                raise HSILoadError(
                    f"Invalid NPZ: missing 'cube' or 'data' key. Found keys: {list(loaded.keys())}"
                )
            
            return LoadedHSI(
                data=data,
                path=npz_path,
                artifact_type=artifact,
                shape=data.shape
            )
        except Exception as e:
            if isinstance(e, HSILoadError):
                raise
            raise HSILoadError(f"Failed to load NPZ: {e}")
    
    if npy_path.exists():
        try:
            data = np.load(npy_path)
            return LoadedHSI(
                data=data,
                path=npy_path,
                artifact_type=artifact,
                shape=data.shape
            )
        except Exception as e:
            raise HSILoadError(f"Failed to load NPY: {e}")
    
    raise HSINotFoundError(
        f"HSI artifact '{artifact}' not found in {from_dir}. "
        f"Expected: {base_name}.npz or {base_name}.npy. "
        f"Run 'spectrapipe run' first to generate outputs."
    )
