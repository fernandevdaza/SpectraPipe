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
    wavelength_nm: np.ndarray | None = None  # (31,) if available


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
        LoadedHSI with data, metadata, and wavelength_nm if available.
    
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
    
    npz_path = from_dir / f"{base_name}.npz"
    
    if npz_path.exists():
        try:
            loaded = np.load(npz_path, allow_pickle=True)
            
            if "cube" in loaded:
                data = loaded["cube"]
            elif "data" in loaded:
                # Legacy schema - no longer supported
                raise HSILoadError(
                    f"Legacy NPZ schema not supported: '{npz_path.name}' uses 'data' key instead of 'cube'. "
                    f"Re-export with current pipeline version using 'spectrapipe run'."
                )
            else:
                raise HSILoadError(
                    f"Invalid NPZ: missing 'cube' key. Found keys: {list(loaded.keys())}"
                )
            
            wavelength_nm = None
            if "wavelength_nm" in loaded:
                wavelength_nm = loaded["wavelength_nm"]
            
            return LoadedHSI(
                data=data,
                path=npz_path,
                artifact_type=artifact,
                shape=data.shape,
                wavelength_nm=wavelength_nm
            )
        except Exception as e:
            if isinstance(e, HSILoadError):
                raise
            raise HSILoadError(f"Failed to load NPZ: {e}")
    
    raise HSINotFoundError(
        f"HSI artifact '{artifact}' not found in {from_dir}. "
        f"Expected: {base_name}.npz. "
        f"Run 'spectrapipe run' first to generate outputs."
    )

