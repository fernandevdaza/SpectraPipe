"""Wavelength utilities for spectral data."""

import json
import csv
from pathlib import Path
from typing import Optional
import numpy as np


class WavelengthError(Exception):
    """Error with wavelength configuration."""
    pass


NUM_BANDS = 31


def load_wavelengths(path: Path) -> np.ndarray:
    """Load wavelengths from CSV or JSON file.
    
    Args:
        path: Path to wavelengths file.
    
    Returns:
        Array of 31 wavelength values in nm.
    
    Raises:
        WavelengthError: If file is invalid or doesn't have 31 values.
    """
    path = Path(path)
    
    if not path.exists():
        raise WavelengthError(f"Wavelengths file not found: {path}")
    
    suffix = path.suffix.lower()
    
    try:
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                values = data
            elif isinstance(data, dict) and "wavelengths" in data:
                values = data["wavelengths"]
            else:
                raise WavelengthError(
                    "Invalid JSON format. Expected list or dict with 'wavelengths' key."
                )
        
        elif suffix == ".csv":
            values = []
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and not row[0].startswith("#"):
                        try:
                            values.append(float(row[0]))
                        except ValueError:
                            continue  # Skip header or invalid row
        
        else:
            raise WavelengthError(
                f"Unsupported wavelengths format: {suffix}. Use .json or .csv"
            )
        
    except (json.JSONDecodeError, csv.Error) as e:
        raise WavelengthError(f"Failed to parse wavelengths file: {e}")
    
    if len(values) != NUM_BANDS:
        raise WavelengthError(
            f"Wavelengths file must have exactly {NUM_BANDS} values, got {len(values)}"
        )
    
    return np.array(values, dtype=np.float32)


def generate_wavelengths(start_nm: float, step_nm: float) -> np.ndarray:
    """Generate wavelength array from start and step.
    
    Args:
        start_nm: Starting wavelength in nm.
        step_nm: Step between bands in nm.
    
    Returns:
        Array of 31 wavelength values.
    """
    return np.array([start_nm + i * step_nm for i in range(NUM_BANDS)], dtype=np.float32)


def get_wavelengths(
    file_path: Optional[Path] = None,
    start_nm: Optional[float] = None,
    step_nm: Optional[float] = None
) -> Optional[np.ndarray]:
    """Get wavelengths from file or parameters.
    
    Args:
        file_path: Path to wavelengths file (optional).
        start_nm: Starting wavelength if generating (optional).
        step_nm: Step between bands if generating (optional).
    
    Returns:
        Wavelength array or None if not specified.
    
    Raises:
        WavelengthError: If configuration is invalid.
    """
    if file_path is not None:
        return load_wavelengths(file_path)
    
    if start_nm is not None and step_nm is not None:
        return generate_wavelengths(start_nm, step_nm)
    
    if start_nm is not None or step_nm is not None:
        raise WavelengthError(
            "Both --wl-start and --wl-step must be provided together"
        )
    
    return None
