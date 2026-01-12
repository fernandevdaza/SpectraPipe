"""Command-line interface for SpectraPipe - HSI Pipeline."""

from pathlib import Path
from typing import Optional
from .pipeline.run import load_config
from .transforms.rgb_to_hsi import rgb_to_hsi
import typer
from rich.console import Console

app = typer.Typer(
    name="SpectraPipe - HSI Pipeline",
    help="A Hyperspectral Image Processing CLI utility!",
    add_completion=False,
)
console = Console()

@app.command()
def run(
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to RGB image",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out", "-o",
        help="Path to output directory",
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    config: Path = typer.Option(
        "configs/defaults.yaml",
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
    no_ensemble: bool = typer.Option(
        False,
        "--no-ensemble",
        help="Disable model ensembling (faster but potentially less accurate)."
    ),
):
    """Run the full HSI processing pipeline."""
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError 

    import time
    from rich.table import Table

    start_time = time.perf_counter()

    cfg = load_config(config)

    if out is None:
        out = input.parent / "output"

    console.print(f"Loading image: {input}")

    try:
        with Image.open(input) as img:
            img.load()
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        console.print("[red]Integrity Error:[/red] Failed to load image data (corrupt or unsupported)")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        raise typer.Exit(1)
    
    rgb = cv2.imread(str(input))
    if rgb is None:
        console.print("[red]Error:[/red] OpenCV failed to load the image (format might not be supported by cv2).")
        raise typer.Exit(1)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if rgb.shape[0] < 32 or rgb.shape[1] < 32:
        console.print(f"[red]Error:[/red] Image is too small ({rgb.shape[1]}x{rgb.shape[0]}). Minimum required is 32x32 pixels.")
        raise typer.Exit(1)

    console.print("Converting RGB → HSI using MST++...")

    use_ensemble = not no_ensemble
    
    try:
        out.mkdir(parents=True, exist_ok=True)
        test_file = out / ".write_check"
        test_file.touch()
        test_file.unlink()

        hsi = rgb_to_hsi(rgb, cfg, ensemble_override=use_ensemble)
        output = out / "hsi_raw_full.npy"
        np.save(output, hsi)
    except PermissionError:
        console.print(f"[red]Permission Error:[/red] Failed to write to output directory: {out}")
        console.print("Check folder permissions or run with 'sudo' (not recommended).")
        raise typer.Exit(1)
    except OSError as e:
        console.print("[red]System Error:[/red] Failed to create/access output directory.")
        console.print(f"Detail: {e}")
        raise typer.Exit(1)

    end_time = time.perf_counter()
    duration = end_time - start_time

    table = Table(title="SpectraPipe Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Image", str(input))
    table.add_row("Output Directory", str(out))
    table.add_row("Output File", str(output.name))
    table.add_row("Config File", str(config))
    table.add_row("Ensemble Enabled", "No" if no_ensemble else "Yes")
    table.add_row("HSI Shape", str(hsi.shape))
    table.add_row("Total Time", f"{duration:.2f}s")

    console.print(table)
    console.print("[green]✓[/green] Pipeline finished successfully.")

    

def main():
    app()

if __name__ == "__main":
    main()