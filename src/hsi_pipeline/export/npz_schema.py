"""NPZ Schema v1 for self-descriptive HSI artifacts."""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import warnings


SCHEMA_VERSION = 1

KEY_CUBE = "cube"
KEY_SCHEMA_VERSION = "schema_version"
KEY_METADATA = "metadata"

KEY_WAVELENGTH_NM = "wavelength_nm"

# Legacy key (for backward compatibility)
KEY_DATA_LEGACY = "data"

EXPECTED_BANDS = 31


class NPZSchemaError(Exception):
    """Error with NPZ schema validation."""
    pass


@dataclass
class NPZMetadata:
    """Metadata for NPZ artifact."""
    schema_version: int = SCHEMA_VERSION
    artifact: str = "raw"  # raw, clean, upscaled_baseline, upscaled_improved
    model_name: str = "MST++"
    input_path: Optional[str] = None
    original_shape: Optional[Tuple[int, int, int]] = None
    fitted_shape: Optional[Tuple[int, int, int]] = None
    cube_shape: Optional[Tuple[int, int, int]] = None
    unit: str = "relative"
    value_range: Optional[Dict[str, float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "1.0.0"
    run_config_ref: Optional[str] = None
    git_commit: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        data = {
            "schema_version": self.schema_version,
            "artifact": self.artifact,
            "model_name": self.model_name,
            "unit": self.unit,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
        }
        
        if self.input_path:
            data["input_path"] = self.input_path
        if self.original_shape:
            data["original_shape"] = list(self.original_shape)
        if self.fitted_shape:
            data["fitted_shape"] = list(self.fitted_shape)
        if self.cube_shape:
            data["cube_shape"] = list(self.cube_shape)
        if self.value_range:
            data["value_range"] = self.value_range
        if self.run_config_ref:
            data["run_config_ref"] = self.run_config_ref
        if self.git_commit:
            data["git_commit"] = self.git_commit
        if self.extra:
            data.update(self.extra)
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "NPZMetadata":
        """Deserialize metadata from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise NPZSchemaError(f"Invalid metadata JSON: {e}")
        
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            artifact=data.get("artifact", "raw"),
            model_name=data.get("model_name", "MST++"),
            input_path=data.get("input_path"),
            original_shape=tuple(data["original_shape"]) if "original_shape" in data else None,
            fitted_shape=tuple(data["fitted_shape"]) if "fitted_shape" in data else None,
            cube_shape=tuple(data["cube_shape"]) if "cube_shape" in data else None,
            unit=data.get("unit", "relative"),
            value_range=data.get("value_range"),
            created_at=data.get("created_at", ""),
            pipeline_version=data.get("pipeline_version", ""),
            run_config_ref=data.get("run_config_ref"),
            git_commit=data.get("git_commit"),
        )


def save_npz_v1(
    path: Path,
    cube: np.ndarray,
    metadata: NPZMetadata,
    wavelength_nm: Optional[np.ndarray] = None
) -> Path:
    """Save HSI cube to NPZ with schema v1.
    
    Args:
        path: Output path.
        cube: HSI cube (H, W, 31) float32.
        metadata: NPZMetadata object.
        wavelength_nm: Optional wavelength array (31,).
    
    Returns:
        Path to saved file.
    
    Raises:
        NPZSchemaError: If validation fails.
    """
    path = Path(path)
    
    # Validate cube
    if cube.ndim != 3:
        raise NPZSchemaError(f"cube must be 3D, got {cube.ndim}D")
    
    if cube.shape[2] != EXPECTED_BANDS:
        raise NPZSchemaError(
            f"cube must have {EXPECTED_BANDS} bands, got {cube.shape[2]}"
        )
    
    if cube.dtype != np.float32:
        cube = cube.astype(np.float32)
    
    metadata.cube_shape = cube.shape
    if metadata.value_range is None:
        metadata.value_range = {
            "min": float(cube.min()),
            "max": float(cube.max())
        }
    
    if wavelength_nm is not None:
        if len(wavelength_nm) != EXPECTED_BANDS:
            raise NPZSchemaError(
                f"wavelength_nm must have {EXPECTED_BANDS} values, got {len(wavelength_nm)}"
            )
        wavelength_nm = wavelength_nm.astype(np.float32)
    
    save_dict = {
        KEY_CUBE: cube,
        KEY_SCHEMA_VERSION: np.array(SCHEMA_VERSION, dtype=np.int32),
        KEY_METADATA: np.array(metadata.to_json()),
    }
    
    if wavelength_nm is not None:
        save_dict[KEY_WAVELENGTH_NM] = wavelength_nm
    
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **save_dict)
    
    return path


@dataclass
class LoadedNPZ:
    """Result of loading NPZ file."""
    cube: np.ndarray
    metadata: Optional[NPZMetadata]
    wavelength_nm: Optional[np.ndarray]
    schema_version: int
    is_legacy: bool
    path: Path


def load_npz_v1(path: Path) -> LoadedNPZ:
    """Load HSI cube from NPZ (v1 or legacy).
    
    Args:
        path: Path to NPZ file.
    
    Returns:
        LoadedNPZ with cube and metadata.
    
    Raises:
        NPZSchemaError: If NPZ is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise NPZSchemaError(f"NPZ file not found: {path}")
    
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        raise NPZSchemaError(f"Failed to load NPZ: {e}")
    
    # Determine which key has the cube
    is_legacy = False
    
    if KEY_CUBE in npz:
        cube = npz[KEY_CUBE]
    elif KEY_DATA_LEGACY in npz:
        cube = npz[KEY_DATA_LEGACY]
        is_legacy = True
        warnings.warn(
            f"Legacy NPZ detected: '{path.name}' uses 'data' instead of 'cube'. "
            "Re-export to update to schema v1.",
            DeprecationWarning
        )
    else:
        raise NPZSchemaError(
            f"Invalid NPZ: missing 'cube' or 'data' key. "
            f"Found keys: {list(npz.keys())}"
        )
    
    # Validate cube shape
    if cube.ndim != 3:
        raise NPZSchemaError(f"Cube must be 3D, got {cube.ndim}D")
    
    if cube.shape[2] != EXPECTED_BANDS:
        raise NPZSchemaError(
            f"Cube must have {EXPECTED_BANDS} bands, got {cube.shape[2]}"
        )
    
    # Load schema version
    schema_version = 0
    if KEY_SCHEMA_VERSION in npz:
        schema_version = int(npz[KEY_SCHEMA_VERSION])
    
    # Load metadata
    metadata = None
    if KEY_METADATA in npz:
        try:
            metadata_str = str(npz[KEY_METADATA])
            metadata = NPZMetadata.from_json(metadata_str)
        except NPZSchemaError:
            pass  # Metadata is optional
    
    # Load wavelengths
    wavelength_nm = None
    if KEY_WAVELENGTH_NM in npz:
        wavelength_nm = npz[KEY_WAVELENGTH_NM]
        if len(wavelength_nm) != EXPECTED_BANDS:
            raise NPZSchemaError(
                f"wavelength_nm must have {EXPECTED_BANDS} values, got {len(wavelength_nm)}"
            )
    
    return LoadedNPZ(
        cube=cube,
        metadata=metadata,
        wavelength_nm=wavelength_nm,
        schema_version=schema_version,
        is_legacy=is_legacy,
        path=path
    )


def validate_npz_schema(path: Path) -> bool:
    """Validate that an NPZ file conforms to schema v1.
    
    Args:
        path: Path to NPZ file.
    
    Returns:
        True if valid.
    
    Raises:
        NPZSchemaError: If validation fails.
    """
    loaded = load_npz_v1(path)
    
    if loaded.is_legacy:
        raise NPZSchemaError("NPZ uses legacy 'data' key instead of 'cube'")
    
    if loaded.schema_version != SCHEMA_VERSION:
        raise NPZSchemaError(
            f"Expected schema_version {SCHEMA_VERSION}, got {loaded.schema_version}"
        )
    
    if loaded.metadata is None:
        raise NPZSchemaError("Missing required 'metadata' key")
    
    return True
