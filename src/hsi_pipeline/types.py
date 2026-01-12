"""Core types and dataclasses for the HSI pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration for the MST++ model."""
    name: str = "mst_plus_plus"
    weights_path: Path = Path("external/MST-plus-plus/test_develop_code/model_zoo/mst_plus_plus.pth")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    ensemble_mode: Literal["mean", "median"] = "mean"
    n_bands: int = 31
    wavelength_range: tuple[int, int] = (400, 700)
    
    def __post_init__(self):
        if isinstance(self.weights_path, str):
            self.weights_path = Path(self.weights_path)

@dataclass
class OutputConfig:
    """Configuration for output paths."""
    runs_dir: Path = Path("runs")
    
    def __post_init__(self):
        if isinstance(self.runs_dir, str):
            self.runs_dir = Path(self.runs_dir)


@dataclass
class RunConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    steps: list[str] = field(default_factory=lambda: ["convert"])
    
    @classmethod
    def from_dict(cls, data: dict | None) -> "RunConfig":
        """Create RunConfig from a dictionary (e.g., from YAML)."""
        if data is None:
            data = {}
            
        def safe_get(d, key, default):
            val = d.get(key)
            return val if val is not None else default

        pipeline_data = safe_get(data, "pipeline", {})
        
        return cls(
            model=ModelConfig(**safe_get(data, "model", {})),
            output=OutputConfig(**safe_get(data, "output", {})),
            steps=pipeline_data.get("steps", ["convert"]),
        )