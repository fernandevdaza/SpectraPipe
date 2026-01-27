"""Pipeline configuration loading and management."""

import warnings
from pathlib import Path

import yaml

from ..types import RunConfig


# Default config template with comments
DEFAULT_CONFIG_TEMPLATE = '''# SpectraPipe Configuration - v2.0
# CLI flags override these values

model:
  name: mst_plus_plus
  weights_path: src/hsi_pipeline/weights/model_zoo/mst_plus_plus.pth
  device: auto  # auto, cuda, cpu
  ensemble: true
  ensemble_mode: mean  # mean, median
  n_bands: 31

fitting:
  multiple: 32
  policy: pad_to_multiple

upscaling:
  enabled: false
  factor: 2
  methods: [baseline, improved]

export:
  format: npz  # npz only
  overwrite: true  # true, false
  default_dir: output  # Default output dir when --out not specified

spectra:
  normalize: none  # none, minmax, zscore, l2
  export_format: json  # json, csv, both

dataset:
  on_error: continue  # continue, abort
  on_annot_error: continue  # continue, abort
'''


def generate_default_config(config_path: Path) -> None:
    """Generate a default config file at the given path.
    
    Args:
        config_path: Path where to create the config file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE)


def load_config(config_path: Path | str) -> RunConfig:
    """Load and validate YAML configuration.
    
    If the config file doesn't exist, it will be generated with defaults.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        RunConfig object with all settings.
    """
    config_path = Path(config_path)
    
    # Auto-generate config if missing
    if not config_path.exists():
        warnings.warn(
            f"Config file not found: {config_path}. "
            f"Generating default config.",
            UserWarning
        )
        generate_default_config(config_path)
    
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML in config file: {config_path}\n"
            f"Error: {e}\n"
            f"Suggestion: Fix the YAML syntax or delete the file to regenerate defaults."
        ) from e
    
    if data is None:
        data = {}
    
    # Warn about unknown top-level keys (forward compatibility)
    known_keys = {"model", "fitting", "upscaling", "export", "spectra", "dataset"}
    unknown_keys = set(data.keys()) - known_keys
    if unknown_keys:
        warnings.warn(
            f"Unknown config keys will be ignored: {unknown_keys}. "
            f"Check spelling or update your config file.",
            UserWarning
        )
    
    return RunConfig.from_dict(data)


def merge_cli_overrides(config: RunConfig, **overrides) -> RunConfig:
    """Apply CLI overrides to config values.
    
    CLI flags have priority over config.yaml values.
    
    Args:
        config: Base configuration from YAML.
        **overrides: CLI overrides as key=value pairs.
            Supported keys:
            - no_ensemble: bool
            - upscale_factor: int | None
            - on_error: str
            - on_annot_error: str
    
    Returns:
        New RunConfig with overrides applied.
    """
    from dataclasses import replace
    
    # Apply model overrides
    if overrides.get("no_ensemble") is True:
        config = replace(config, model=replace(config.model, ensemble=False))
    
    # Apply upscaling overrides
    if overrides.get("upscale_factor") is not None:
        config = replace(
            config, 
            upscaling=replace(
                config.upscaling, 
                enabled=True, 
                factor=overrides["upscale_factor"]
            )
        )
    
    # Apply dataset policy overrides
    if overrides.get("on_error") is not None:
        config = replace(
            config,
            dataset=replace(config.dataset, on_error=overrides["on_error"])
        )
    
    if overrides.get("on_annot_error") is not None:
        config = replace(
            config,
            dataset=replace(config.dataset, on_annot_error=overrides["on_annot_error"])
        )
    
    return config