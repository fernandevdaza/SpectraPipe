from pathlib import Path
from ..types import RunConfig
import yaml

def load_config(config_path: Path | str) -> RunConfig:
    """Load and validate YAML configuration.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        RunConfig object with all settings.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return RunConfig.from_dict(data)