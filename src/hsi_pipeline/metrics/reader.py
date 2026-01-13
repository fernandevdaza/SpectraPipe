"""Metrics JSON reader with validation."""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Result of reading metrics.json."""
    data: dict
    warnings: list[str]


class MetricsNotFoundError(Exception):
    """Raised when metrics.json is not found."""
    pass


class MetricsCorruptError(Exception):
    """Raised when metrics.json is invalid JSON."""
    pass


def read_metrics(metrics_path: Path) -> MetricsResult:
    """Read and validate metrics.json.
    
    Args:
        metrics_path: Path to metrics.json file.
    
    Returns:
        MetricsResult with data and any warnings.
    
    Raises:
        MetricsNotFoundError: If file doesn't exist.
        MetricsCorruptError: If JSON is invalid.
    """
    if not metrics_path.exists():
        raise MetricsNotFoundError(
            f"metrics.json not found at: {metrics_path}"
        )
    
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MetricsCorruptError(
            f"metrics.json is corrupt (invalid JSON): {e}"
        )
    
    if not isinstance(data, dict):
        raise MetricsCorruptError(
            "metrics.json has invalid structure (expected object)"
        )
    
    warnings = validate_metrics(data)
    
    return MetricsResult(data=data, warnings=warnings)


def validate_metrics(data: dict) -> list[str]:
    """Validate metrics data and return warnings for issues.
    
    Args:
        data: Parsed metrics dictionary.
    
    Returns:
        List of warning messages.
    """
    warnings = []
    
    # Check expected fields and types
    expected_fields = {
        "hsi_shape": list,
        "n_bands": (int, type(None)),
        "execution_time_seconds": (int, float),
        "ensemble_enabled": bool,
        "timestamp": str,
    }
    
    for field, expected_type in expected_fields.items():
        if field not in data:
            warnings.append(f"Missing field: {field}")
        elif not isinstance(data[field], expected_type):
            warnings.append(
                f"Unexpected type for '{field}': expected {expected_type}, got {type(data[field]).__name__}"
            )
    
    # Check optional sections and suggest how to generate them
    if "raw_separability" not in data:
        warnings.append(
            "Missing: separability metrics → run with `--roi-mask <mask>` to generate"
        )
    
    if "clean_separability" not in data and "raw_clean_sam" not in data:
        warnings.append(
            "Missing: clean metrics → run with `--roi-mask <mask> --clean` to generate"
        )
    
    if "upscale_factor" not in data and "upscaled_size" not in data:
        warnings.append(
            "Missing: upscaling metrics → run with `--upscale` to generate"
        )
    
    return warnings
