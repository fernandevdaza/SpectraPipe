"""Export spectral signatures to CSV and JSON."""

import json
import csv
from pathlib import Path
from typing import Literal, Optional, List
import numpy as np

from .extractor import SpectralSignature


def get_export_filename(
    signature: SpectralSignature,
    format: str,
    mode: str = "single"
) -> str:
    """Generate filename for exported signature.
    
    Args:
        signature: The spectral signature.
        format: Export format ('csv' or 'json').
        mode: 'single', 'pixel', 'roi', or 'batch'.
    
    Returns:
        Generated filename.
    """
    if mode == "batch":
        return f"spectra_{signature.artifact}_batch_pixels.{format}"
    
    if signature.source == "pixel":
        name = f"spectra_{signature.artifact}_pixel_{signature.pixel_x}_{signature.pixel_y}"
    else:
        name = f"spectra_{signature.artifact}_roi_{signature.roi_aggregation}"
    
    return f"{name}.{format}"


def export_csv(
    signature: SpectralSignature,
    output_dir: Path,
    filename: str | None = None,
    wavelengths: Optional[np.ndarray] = None,
    normalize_mode: str = "none",
    source_path: Optional[str] = None
) -> Path:
    """Export spectral signature to CSV file.
    
    Args:
        signature: Spectral signature to export.
        output_dir: Directory to write file.
        filename: Optional custom filename.
        wavelengths: Optional wavelength array.
        normalize_mode: Normalization applied.
        source_path: Path to source HSI.
    
    Returns:
        Path to exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = get_export_filename(signature, "csv")
    
    filepath = output_dir / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        if wavelengths is not None:
            writer.writerow(["band", "wavelength_nm", "value"])
        else:
            writer.writerow(["band", "value"])
        
        for i, value in enumerate(signature.values):
            if wavelengths is not None:
                writer.writerow([i + 1, float(wavelengths[i]), float(value)])
            else:
                writer.writerow([i + 1, float(value)])
    
    return filepath


def export_json(
    signature: SpectralSignature,
    output_dir: Path,
    filename: str | None = None,
    wavelengths: Optional[np.ndarray] = None,
    normalize_mode: str = "none",
    source_path: Optional[str] = None
) -> Path:
    """Export spectral signature to JSON file.
    
    Args:
        signature: Spectral signature to export.
        output_dir: Directory to write file.
        filename: Optional custom filename.
        wavelengths: Optional wavelength array.
        normalize_mode: Normalization applied.
        source_path: Path to source HSI.
    
    Returns:
        Path to exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = get_export_filename(signature, "json")
    
    filepath = output_dir / filename
    
    data = {
        "artifact": signature.artifact,
        "source": signature.source,
        "mode": signature.source,
        "bands": signature.num_bands,
        "values": [float(v) for v in signature.values],
        "normalize": normalize_mode,
    }
    
    if wavelengths is not None:
        data["wavelength_nm"] = [float(w) for w in wavelengths]
    
    if source_path:
        data["source_path"] = source_path
    
    if signature.source == "pixel":
        data["pixel"] = {"x": signature.pixel_x, "y": signature.pixel_y}
    else:
        data["roi"] = {
            "aggregation": signature.roi_aggregation,
            "pixel_count": signature.roi_pixel_count
        }
        data["roi_agg"] = signature.roi_aggregation
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def export_batch_csv(
    signatures: List[SpectralSignature],
    output_dir: Path,
    artifact: str,
    wavelengths: Optional[np.ndarray] = None,
    normalize_mode: str = "none"
) -> Path:
    """Export batch of signatures to CSV file.
    
    Args:
        signatures: List of SpectralSignature objects.
        output_dir: Directory to write file.
        artifact: Artifact type.
        wavelengths: Optional wavelength array.
        normalize_mode: Normalization applied.
    
    Returns:
        Path to exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"spectra_{artifact}_batch_pixels.csv"
    filepath = output_dir / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Build header
        header = ["x", "y"]
        if wavelengths is not None:
            header.extend([f"band_{i+1}_{wavelengths[i]:.1f}nm" for i in range(31)])
        else:
            header.extend([f"band_{i+1}" for i in range(31)])
        
        writer.writerow(header)
        
        for sig in signatures:
            row = [sig.pixel_x, sig.pixel_y] + [float(v) for v in sig.values]
            writer.writerow(row)
    
    return filepath


def export_batch_json(
    signatures: List[SpectralSignature],
    output_dir: Path,
    artifact: str,
    wavelengths: Optional[np.ndarray] = None,
    normalize_mode: str = "none",
    failed: Optional[List[dict]] = None
) -> Path:
    """Export batch of signatures to JSON file.
    
    Args:
        signatures: List of SpectralSignature objects.
        output_dir: Directory to write file.
        artifact: Artifact type.
        wavelengths: Optional wavelength array.
        normalize_mode: Normalization applied.
        failed: Optional list of failed extractions.
    
    Returns:
        Path to exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"spectra_{artifact}_batch_pixels.json"
    filepath = output_dir / filename
    
    data = {
        "artifact": artifact,
        "mode": "batch",
        "bands": 31,
        "normalize": normalize_mode,
        "total_signatures": len(signatures),
        "signatures": []
    }
    
    if wavelengths is not None:
        data["wavelength_nm"] = [float(w) for w in wavelengths]
    
    for sig in signatures:
        data["signatures"].append({
            "x": sig.pixel_x,
            "y": sig.pixel_y,
            "values": [float(v) for v in sig.values]
        })
    
    if failed:
        data["failed"] = failed
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def export_signature(
    signature: SpectralSignature,
    output_dir: Path,
    format: Literal["csv", "json", "both"] = "both",
    wavelengths: Optional[np.ndarray] = None,
    normalize_mode: str = "none",
    source_path: Optional[str] = None
) -> list[Path]:
    """Export spectral signature to file(s).
    
    Args:
        signature: Spectral signature to export.
        output_dir: Directory to write files.
        format: Export format ('csv', 'json', or 'both').
        wavelengths: Optional wavelength array.
        normalize_mode: Normalization applied.
        source_path: Path to source HSI.
    
    Returns:
        List of exported file paths.
    """
    paths = []
    
    if format in ("csv", "both"):
        paths.append(export_csv(
            signature, output_dir,
            wavelengths=wavelengths,
            normalize_mode=normalize_mode,
            source_path=source_path
        ))
    
    if format in ("json", "both"):
        paths.append(export_json(
            signature, output_dir,
            wavelengths=wavelengths,
            normalize_mode=normalize_mode,
            source_path=source_path
        ))
    
    return paths
