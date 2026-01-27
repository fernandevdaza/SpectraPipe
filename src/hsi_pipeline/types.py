"""Core types and dataclasses for the HSI pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for the MST++ model."""
    name: str = "mst_plus_plus"
    weights_path: Path = Path("src/hsi_pipeline/weights/model_zoo/mst_plus_plus.pth")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    ensemble: bool = True
    ensemble_mode: Literal["mean", "median"] = "mean"
    n_bands: int = 31
    
    def __post_init__(self):
        if isinstance(self.weights_path, str):
            self.weights_path = Path(self.weights_path)


@dataclass
class FittingConfig:
    """Configuration for input fitting."""
    multiple: int = 32
    policy: str = "pad_to_multiple"


@dataclass
class UpscalingConfig:
    """Configuration for HSI upscaling."""
    enabled: bool = False
    factor: int = 2
    methods: list[str] = field(default_factory=lambda: ["baseline", "improved"])


@dataclass
class ExportConfig:
    """Configuration for artifact export."""
    format: Literal["npz"] = "npz"
    overwrite: bool = True
    default_dir: str = "output"  # Default output dir when --out not specified


@dataclass
class SpectraConfig:
    """Configuration for spectral signature extraction."""
    normalize: Literal["none", "minmax", "zscore", "l2"] = "none"
    export_format: Literal["json", "csv", "both"] = "json"


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    on_error: Literal["continue", "abort"] = "continue"
    on_annot_error: Literal["continue", "abort"] = "continue"


@dataclass
class RunConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    spectra: SpectraConfig = field(default_factory=SpectraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    @classmethod
    def from_dict(cls, data: dict | None) -> "RunConfig":
        """Create RunConfig from a dictionary (e.g., from YAML)."""
        if data is None:
            data = {}
        
        def safe_get(d: dict, key: str, default_factory):
            """Get value from dict, filtering None values from nested dicts."""
            val = d.get(key)
            if val is None:
                return default_factory()
            if isinstance(val, dict):
                # Filter out None values to let defaults apply
                filtered = {k: v for k, v in val.items() if v is not None}
                return default_factory(**filtered) if filtered else default_factory()
            return val
        
        return cls(
            model=safe_get(data, "model", ModelConfig),
            fitting=safe_get(data, "fitting", FittingConfig),
            upscaling=safe_get(data, "upscaling", UpscalingConfig),
            export=safe_get(data, "export", ExportConfig),
            spectra=safe_get(data, "spectra", SpectraConfig),
            dataset=safe_get(data, "dataset", DatasetConfig),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML export."""
        return {
            "model": {
                "name": self.model.name,
                "weights_path": str(self.model.weights_path),
                "device": self.model.device,
                "ensemble": self.model.ensemble,
                "ensemble_mode": self.model.ensemble_mode,
                "n_bands": self.model.n_bands,
            },
            "fitting": {
                "multiple": self.fitting.multiple,
                "policy": self.fitting.policy,
            },
            "upscaling": {
                "enabled": self.upscaling.enabled,
                "factor": self.upscaling.factor,
                "methods": self.upscaling.methods,
            },
            "export": {
                "format": self.export.format,
                "overwrite": self.export.overwrite,
                "default_dir": self.export.default_dir,
            },
            "spectra": {
                "normalize": self.spectra.normalize,
                "export_format": self.spectra.export_format,
            },
            "dataset": {
                "on_error": self.dataset.on_error,
                "on_annot_error": self.dataset.on_annot_error,
            },
        }