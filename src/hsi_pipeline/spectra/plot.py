"""Plot generation for spectral signatures."""

from pathlib import Path
from typing import Optional, List
import numpy as np


def plot_signature(
    values: np.ndarray,
    output_path: Path,
    wavelengths: Optional[np.ndarray] = None,
    title: str = "Spectral Signature",
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Reflectance"
) -> Path:
    """Generate plot of spectral signature.
    
    Args:
        values: 1D array of spectral values.
        output_path: Path to save PNG.
        wavelengths: Optional wavelength array for x-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    
    Returns:
        Path to saved plot.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if wavelengths is not None:
        x = wavelengths
    else:
        x = np.arange(1, len(values) + 1)
        xlabel = "Band"
    
    ax.plot(x, values, 'b-', linewidth=1.5, marker='o', markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    return output_path


def plot_batch_signatures(
    signatures: List,
    wavelengths: Optional[np.ndarray],
    output_dir: Path,
    artifact: str,
    max_plots: int = 10
) -> List[Path]:
    """Generate plots for batch of signatures.
    
    Args:
        signatures: List of SpectralSignature objects.
        wavelengths: Optional wavelength array.
        output_dir: Directory to save plots.
        artifact: Artifact type for filename.
        max_plots: Maximum number of plots to generate.
    
    Returns:
        List of paths to generated plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, sig in enumerate(signatures[:max_plots]):
        filename = f"spectra_{artifact}_pixel_{sig.pixel_x}_{sig.pixel_y}.png"
        plot_path = output_dir / filename
        
        title = f"Spectral Signature - Pixel ({sig.pixel_x}, {sig.pixel_y})"
        plot_signature(sig.values, plot_path, wavelengths, title=title)
        paths.append(plot_path)
    
    return paths
