"""Manifest parser for dataset processing."""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union
import glob as glob_module


class ManifestError(Exception):
    """Base exception for manifest errors."""
    pass


class ManifestNotFoundError(ManifestError):
    """Manifest file not found."""
    pass


class ManifestParseError(ManifestError):
    """Failed to parse manifest."""
    pass


class ManifestValidationError(ManifestError):
    """Manifest validation failed."""
    pass


@dataclass
class Sample:
    """A single sample in the dataset."""
    id: str
    image: str
    roi_mask: Optional[str] = None
    
    annotation: Optional[str] = None  # Path to annotation file
    annotation_type: Optional[str] = None  # 'voc' or 'coco'
    
    image_resolved: Optional[Path] = field(default=None, repr=False)
    roi_mask_resolved: Optional[Path] = field(default=None, repr=False)
    annotation_resolved: Optional[Path] = field(default=None, repr=False)


@dataclass
class Manifest:
    """Parsed and validated manifest."""
    root: Path
    samples: List[Sample]
    source_path: Path
    pattern: Optional[str] = None


def parse_manifest(path: Union[str, Path]) -> Manifest:
    """Parse a YAML or JSON manifest file.
    
    Args:
        path: Path to manifest file (.yaml, .yml, or .json)
    
    Returns:
        Parsed Manifest object
    
    Raises:
        ManifestNotFoundError: If manifest file doesn't exist
        ManifestParseError: If file can't be parsed
        ManifestValidationError: If manifest is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise ManifestNotFoundError(f"Manifest not found: {path}")
    
    if not path.is_file():
        raise ManifestNotFoundError(f"Manifest is not a file: {path}")
    
    suffix = path.suffix.lower()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if suffix in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            elif suffix == '.json':
                data = json.load(f)
            else:
                raise ManifestParseError(
                    f"Unsupported manifest format: {suffix}. Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        raise ManifestParseError(f"Invalid YAML syntax: {e}")
    except json.JSONDecodeError as e:
        raise ManifestParseError(f"Invalid JSON syntax at line {e.lineno}: {e.msg}")
    
    if not isinstance(data, dict):
        raise ManifestValidationError("Manifest must be a dictionary/object")
    
    return _validate_manifest(data, path)


def _validate_manifest(data: dict, source_path: Path) -> Manifest:
    """Validate manifest data and return Manifest object."""
    
    if 'root' not in data:
        raise ManifestValidationError("Missing required field: 'root'")
    
    root = Path(data['root'])
    if not root.is_absolute():
        root = source_path.parent / root
    root = root.resolve()
    
    if not root.exists():
        raise ManifestValidationError(f"Root directory not found: {root}")
    
    if not root.is_dir():
        raise ManifestValidationError(f"Root is not a directory: {root}")
    
    has_samples = 'samples' in data
    has_pattern = 'pattern' in data
    
    if not has_samples and not has_pattern:
        raise ManifestValidationError(
            "Manifest must have either 'samples' list or 'pattern' for discovery"
        )
    
    if has_samples and has_pattern:
        raise ManifestValidationError(
            "Manifest cannot have both 'samples' and 'pattern'. Choose one."
        )
    
    samples = []
    pattern = None
    
    if has_samples:
        samples = _parse_samples(data['samples'], root)
    else:
        pattern = data['pattern']
        samples = _discover_samples(pattern, root)
    
    _check_duplicate_ids(samples)
    
    return Manifest(
        root=root,
        samples=samples,
        source_path=source_path.resolve(),
        pattern=pattern
    )


def _parse_samples(samples_data: list, root: Path) -> List[Sample]:
    """Parse samples list from manifest."""
    if not isinstance(samples_data, list):
        raise ManifestValidationError("'samples' must be a list")
    
    samples = []
    for i, item in enumerate(samples_data):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"Sample {i}: must be a dictionary")
        
        if 'id' not in item:
            raise ManifestValidationError(f"Sample {i}: missing required field 'id'")
        
        if 'image' not in item:
            raise ManifestValidationError(
                f"Sample '{item.get('id', i)}': missing required field 'image'"
            )
        
        sample = Sample(
            id=str(item['id']),
            image=item['image'],
            roi_mask=item.get('roi_mask'),
            annotation=item.get('annotation'),
            annotation_type=item.get('annotation_type')
        )
        
        sample.image_resolved = _resolve_path(sample.image, root)
        if sample.roi_mask:
            sample.roi_mask_resolved = _resolve_path(sample.roi_mask, root)
        if sample.annotation:
            sample.annotation_resolved = _resolve_path(sample.annotation, root)
        
        samples.append(sample)
    
    return samples


def _discover_samples(pattern: str, root: Path) -> List[Sample]:
    """Discover samples using glob pattern."""
    if not isinstance(pattern, str):
        raise ManifestValidationError("'pattern' must be a string")
    
    full_pattern = str(root / pattern)
    matches = sorted(glob_module.glob(full_pattern))
    
    if not matches:
        raise ManifestValidationError(
            f"Pattern '{pattern}' matched no files in {root}"
        )
    
    samples = []
    for match_path in matches:
        path = Path(match_path)
        sample_id = path.stem
        
        sample = Sample(
            id=sample_id,
            image=str(path.relative_to(root)),
            image_resolved=path.resolve()
        )
        samples.append(sample)
    
    return samples


def _resolve_path(rel_path: str, root: Path) -> Path:
    """Resolve a path relative to root, or return absolute as-is."""
    p = Path(rel_path)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _check_duplicate_ids(samples: List[Sample]) -> None:
    """Check for duplicate sample IDs."""
    seen = set()
    for sample in samples:
        if sample.id in seen:
            raise ManifestValidationError(f"Duplicate sample ID: '{sample.id}'")
        seen.add(sample.id)
