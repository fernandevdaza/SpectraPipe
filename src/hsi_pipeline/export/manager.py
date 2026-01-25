"""Export manager for consistent artifact output."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Literal


ARTIFACT_NAMES = {
    "hsi_raw": "hsi_raw_full",
    "hsi_clean": "hsi_clean_full",
    "hsi_upscaled_baseline": "hsi_upscaled_baseline",
    "hsi_upscaled_improved": "hsi_upscaled_improved",
    "roi_mask": "roi_mask",
    "metrics": "metrics",
    "run_config": "run_config",
}


@dataclass
class ExportManager:
    """Centralizes artifact export with consistent naming and policies.
    
    Attributes:
        out_dir: Output directory path.
        format: Default format for array exports ('npz' or 'npy').
        overwrite: Whether to overwrite existing files.
    """
    out_dir: Path
    format: Literal["npz", "npy"] = "npz"
    overwrite: bool = True
    _exported: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
    
    def prepare_directory(self) -> None:
        """Create output directory if it doesn't exist.
        
        Raises:
            NotADirectoryError: If path exists but is not a directory.
            PermissionError: If directory cannot be created/accessed.
        """
        if self.out_dir.exists():
            if not self.out_dir.is_dir():
                raise NotADirectoryError(
                    f"Output path exists but is not a directory: {self.out_dir}"
                )
        else:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = self.out_dir / ".write_check"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise PermissionError(
                f"Cannot write to output directory: {self.out_dir}"
            )
    
    def get_path(self, artifact_key: str, ext: str | None = None) -> Path:
        """Get the full path for an artifact.
        
        Args:
            artifact_key: Key from ARTIFACT_NAMES or custom name.
            ext: Extension override (default uses self.format for arrays).
        
        Returns:
            Full path to the artifact file.
        """
        name = ARTIFACT_NAMES.get(artifact_key, artifact_key)
        
        if ext is None:
            if artifact_key in ["metrics", "run_config"]:
                ext = "json"
            elif artifact_key == "roi_mask":
                ext = "png"
            else:
                ext = self.format
        
        return self.out_dir / f"{name}.{ext}"
    
    def export_array(self, artifact_key: str, data: np.ndarray) -> Path:
        """Export a numpy array to file.
        
        Args:
            artifact_key: Key identifying the artifact type.
            data: Numpy array to export.
        
        Returns:
            Path to the exported file.
        """
        path = self.get_path(artifact_key)
        
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Artifact already exists: {path}")
        
        # Validate for NaN/inf
        if np.issubdtype(data.dtype, np.floating):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0 or inf_count > 0:
                import warnings
                warnings.warn(
                    f"Array '{artifact_key}' contains {nan_count} NaN and {inf_count} Inf values",
                    RuntimeWarning
                )
        
        if self.format == "npz":
            np.savez_compressed(path, data=data)
        else:
            np.save(path, data)
        
        self._exported.append(path.name)
        return path
    
    def export_json(self, artifact_key: str, data: dict) -> Path:
        """Export a dictionary to JSON file.
        
        Args:
            artifact_key: Key identifying the artifact type.
            data: Dictionary to export.
        
        Returns:
            Path to the exported file.
        """
        path = self.get_path(artifact_key, ext="json")
        
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Artifact already exists: {path}")
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        self._exported.append(path.name)
        return path
    
    def export_metrics(
        self,
        hsi_shape: tuple,
        execution_time: float,
        ensemble_enabled: bool = True,
        extra: dict | None = None
    ) -> Path:
        """Export standardized metrics.json.
        
        Args:
            hsi_shape: Shape of the output HSI cube.
            execution_time: Total execution time in seconds.
            ensemble_enabled: Whether ensemble was used.
            extra: Additional metrics to include.
        
        Returns:
            Path to metrics.json.
        """
        metrics = {
            "hsi_shape": list(hsi_shape),
            "n_bands": hsi_shape[2] if len(hsi_shape) > 2 else None,
            "execution_time_seconds": round(execution_time, 3),
            "ensemble_enabled": ensemble_enabled,
            "timestamp": datetime.now().isoformat(),
        }
        
        if extra:
            metrics.update(extra)
        
        return self.export_json("metrics", metrics)
    
    def export_run_config(
        self,
        input_path: str,
        config_path: str,
        fitting_info: dict,
        pipeline_version: str = "0.1.0",
        extra: dict | None = None
    ) -> Path:
        """Export standardized run_config.json.
        
        Args:
            input_path: Path to input image.
            config_path: Path to config YAML.
            fitting_info: Input fitting metadata.
            pipeline_version: Pipeline version string.
            extra: Additional config to include.
        
        Returns:
            Path to run_config.json.
        """
        config = {
            "pipeline_version": pipeline_version,
            "timestamp": datetime.now().isoformat(),
            "input_path": str(input_path),
            "config_path": str(config_path),
            "output_dir": str(self.out_dir),
            "fitting": fitting_info,
        }
        
        if extra:
            config.update(extra)
        
        return self.export_json("run_config", config)
    
    def list_exported(self) -> list[str]:
        """Return list of exported artifact filenames."""
        return self._exported.copy()
    
    def cleanup_partial(self) -> list[str]:
        """Remove any partially exported artifacts.
        
        Called on pipeline failure to avoid leaving corrupted files.
        
        Returns:
            List of removed file names.
        """
        removed = []
        for filename in self._exported:
            path = self.out_dir / filename
            if path.exists():
                try:
                    path.unlink()
                    removed.append(filename)
                except OSError:
                    pass  # Best effort cleanup
        self._exported.clear()
        return removed
