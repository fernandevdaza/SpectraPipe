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
    
    def export_array(
        self,
        artifact_key: str,
        data: np.ndarray,
        input_path: str | None = None,
        original_shape: tuple | None = None,
        fitted_shape: tuple | None = None,
        wavelength_nm: np.ndarray | None = None,
        pipeline_version: str = "1.0.0",
    ) -> Path:
        """Export a numpy array to file.
        
        For HSI artifacts (hsi_raw, hsi_clean, etc.), uses NPZ Schema v1.
        
        Args:
            artifact_key: Key identifying the artifact type.
            data: Numpy array to export.
            input_path: Original input image path (for metadata).
            original_shape: Original image shape (for metadata).
            fitted_shape: Fitted image shape (for metadata).
            wavelength_nm: Optional wavelength array (31,).
            pipeline_version: Pipeline version string.
        
        Returns:
            Path to the exported file.
        """
        path = self.get_path(artifact_key)
        
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Artifact already exists: {path}")
        
        if np.issubdtype(data.dtype, np.floating):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0 or inf_count > 0:
                import warnings
                warnings.warn(
                    f"Array '{artifact_key}' contains {nan_count} NaN and {inf_count} Inf values",
                    RuntimeWarning
                )
        
        is_hsi_artifact = artifact_key.startswith("hsi_")
        
        if self.format == "npz" and is_hsi_artifact and data.ndim == 3:
            from .npz_schema import save_npz_v1, NPZMetadata
            
            artifact_type_map = {
                "hsi_raw": "raw",
                "hsi_clean": "clean",
                "hsi_upscaled_baseline": "upscaled_baseline",
                "hsi_upscaled_improved": "upscaled_improved",
            }
            artifact_type = artifact_type_map.get(artifact_key, "raw")
            
            metadata = NPZMetadata(
                artifact=artifact_type,
                input_path=input_path,
                original_shape=original_shape,
                fitted_shape=fitted_shape,
                pipeline_version=pipeline_version,
            )
            
            save_npz_v1(path, data, metadata, wavelength_nm)
        
        elif self.format == "npz":
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
                    pass
        self._exported.clear()
        return removed
    
    def mark_skipped(self, artifact_key: str, reason: str) -> None:
        """Mark an artifact as skipped with reason.
        
        Args:
            artifact_key: Key identifying the artifact.
            reason: Reason why it was skipped.
        """
        if not hasattr(self, '_skipped'):
            self._skipped = []
        self._skipped.append({"artifact": artifact_key, "reason": reason})
    
    def list_skipped(self) -> list[dict]:
        """Return list of skipped artifacts with reasons."""
        return getattr(self, '_skipped', []).copy()
    
    def export_roi(
        self,
        mask: "np.ndarray",
        source_path: str | None = None,
        export_as: str = "png"
    ) -> Path:
        """Export ROI mask file.
        
        Args:
            mask: Binary ROI mask (H, W).
            source_path: Original path (for ref export).
            export_as: Export format ('png', 'npy', 'npz', 'ref').
        
        Returns:
            Path to exported file.
        """
        if export_as == "ref":
            import hashlib
            ref_data = {
                "source_path": str(source_path) if source_path else None,
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "pixel_count": int(mask.sum()),
                "coverage": float(mask.sum() / mask.size),
            }
            if source_path:
                try:
                    with open(source_path, "rb") as f:
                        ref_data["hash_md5"] = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    pass
            return self.export_json("roi_mask_ref", ref_data)
        
        elif export_as == "png":
            from PIL import Image
            path = self.get_path("roi_mask", ext="png")
            if path.exists() and not self.overwrite:
                raise FileExistsError(f"Artifact already exists: {path}")
            img = Image.fromarray((mask * 255).astype(np.uint8))
            img.save(path)
            self._exported.append(path.name)
            return path
        
        elif export_as in ("npy", "npz"):
            path = self.get_path("roi_mask", ext=export_as)
            if path.exists() and not self.overwrite:
                raise FileExistsError(f"Artifact already exists: {path}")
            if export_as == "npz":
                np.savez_compressed(path, mask=mask)
            else:
                np.save(path, mask)
            self._exported.append(path.name)
            return path
        
        else:
            raise ValueError(f"Unknown ROI export format: {export_as}")
    
    def get_export_summary(self) -> dict:
        """Get summary of exported and skipped artifacts.
        
        Returns:
            Dict with 'exported' and 'skipped' lists.
        """
        return {
            "exported": self.list_exported(),
            "skipped": self.list_skipped(),
            "exported_count": len(self._exported),
            "skipped_count": len(getattr(self, '_skipped', [])),
        }
    
    def log_export_summary(self, console) -> None:
        """Log export summary to console.
        
        Args:
            console: Rich console for output.
        """
        summary = self.get_export_summary()
        
        if summary["exported"]:
            console.print("[green]Exported artifacts:[/green]")
            for name in summary["exported"]:
                console.print(f"  ✓ {name}")
        
        if summary["skipped"]:
            console.print("[yellow]Skipped artifacts:[/yellow]")
            for item in summary["skipped"]:
                console.print(f"  ⊘ {item['artifact']}: {item['reason']}")

