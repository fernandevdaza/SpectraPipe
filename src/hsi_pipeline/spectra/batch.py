"""Batch extraction utilities."""

import csv
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

from .extractor import extract_pixel, SpectralSignature, CoordinateError


class BatchError(Exception):
    """Error in batch extraction."""
    pass


@dataclass
class BatchResult:
    """Result of batch extraction."""
    signatures: List[SpectralSignature]
    failed: List[dict]  # {x, y, error}
    total: int
    success_count: int
    fail_count: int


def load_pixels_file(path: Path) -> List[Tuple[int, int]]:
    """Load pixel coordinates from CSV file.
    
    Args:
        path: Path to CSV file with x,y columns.
    
    Returns:
        List of (x, y) tuples.
    
    Raises:
        BatchError: If file is invalid or can't be parsed.
    """
    path = Path(path)
    
    if not path.exists():
        raise BatchError(f"Pixels file not found: {path}")
    
    pixels = []
    
    try:
        with open(path) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row or row[0].startswith("#"):
                    continue
                
                # Skip header
                if i == 0 and row[0].lower() in ("x", "col", "column"):
                    continue
                
                if len(row) < 2:
                    raise BatchError(f"Row {i+1}: Expected x,y columns, got {len(row)} values")
                
                try:
                    x = int(row[0].strip())
                    y = int(row[1].strip())
                    pixels.append((x, y))
                except ValueError:
                    raise BatchError(f"Row {i+1}: Invalid coordinates '{row[0]},{row[1]}'")
    
    except csv.Error as e:
        raise BatchError(f"Failed to parse pixels file: {e}")
    
    if not pixels:
        raise BatchError("Pixels file is empty (no valid coordinates)")
    
    return pixels


def extract_batch(
    hsi: np.ndarray,
    pixels: List[Tuple[int, int]],
    artifact_type: str = "raw",
    fail_fast: bool = False
) -> BatchResult:
    """Extract spectral signatures for multiple pixels.
    
    Args:
        hsi: HSI cube (H, W, C).
        pixels: List of (x, y) coordinates.
        artifact_type: Source artifact type.
        fail_fast: If True, stop on first error. If False, continue and report.
    
    Returns:
        BatchResult with signatures and failures.
    """
    signatures = []
    failed = []
    
    for x, y in pixels:
        try:
            sig = extract_pixel(hsi, x, y, artifact_type)
            signatures.append(sig)
        except CoordinateError as e:
            failed.append({"x": x, "y": y, "error": str(e)})
            if fail_fast:
                break
    
    return BatchResult(
        signatures=signatures,
        failed=failed,
        total=len(pixels),
        success_count=len(signatures),
        fail_count=len(failed)
    )
