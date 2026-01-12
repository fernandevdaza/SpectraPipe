"""Command-line interface for SpectraPipe - HSI Pipeline."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
    )
):
    """Run the full HSI processing pipeline."""
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError 
    import shutil

    if out is None:
        out = input.parent / "output"

    console.print(f"Loading image: {input}")

    try:
        with Image.open(input) as img:
            img.load()
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        console.print(f"[red]Integrity Error:[/red] Failed to load image data (corrupt or unsupported)")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        raise typer.Exit(1)
    
    rgb = cv2.imread(str(input))
    if rgb is None:
        console.print(f"[red]Error:[/red] OpenCV failed to load the image (format might not be supported by cv2).")
        raise typer.Exit(1)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    console.print("Converting RGB → HSI using MST++...")

    try:
        out.mkdir(parents=True, exist_ok=True)
        test_file = out / ".write_check"
        test_file.touch()
        test_file.unlink()

        # TODO: implement actual HSI conversion
        # dummy hsi cube
        hsi = np.zeros((rgb.shape[0], rgb.shape[1], 3))
        output = out / "hsi_raw_full.npy"
        np.save(output, hsi)
    except PermissionError:
        console.print(f"[red]Permission Error:[/red] Failed to write to output directory: {out}")
        console.print("Check folder permissions or run with 'sudo' (not recommended).")
        raise typer.Exit(1)
    except OSError as e:
        console.print(f"[red]System Error:[/red] Failed to create/access output directory.")
        console.print(f"Detail: {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Saved HSI cube to: [bold]{output}[/bold]")
    console.print(f"  Shape: {hsi.shape}")

    

def main():
    app()

if __name__ == "__main":
    main()