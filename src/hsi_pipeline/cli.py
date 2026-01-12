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
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration YAML file"
    ),
    samples: Optional[str] = typer.Option(
        None,
        "--samples", "-s",
        help="Comma separated list of sample IDs to process"
    ),
    dataset: str = typer.Option(
        "all",
        "--dataset", "-d",
        help="Dataset to process"
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels", "-l",
        help="Comma separated list of labels to include"
    ),
):
    """Run the full HSI processing pipeline."""

    samples_filter = samples.split(",") if samples else None
    labels_filter = labels.split(",") if labels else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)

        def call_back(current: int, total: int, sample_id: str):
            progress.update(task, description=f"Processing {sample_id} ({current}/{total})")


    console.print(f"\n[green]✓[/green] Pipeline complete!")

@app.command()
def convert(
    input: str = typer.Argument(..., help="Path to input RGB image"),
    output: str = typer.Argument(..., help="Path to output HSI file (.npz)"),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration YAML file"
    ),
    no_ensemble: bool = typer.Option(
        False,
        "--no-ensemble",
        help="Disable test time augmentation (faster but less accurate)"
    )
):
    """Convert a single RGB image to HSI"""
    pass

@app.command()
def upscale(
    input: str = typer.Argument(..., help="Path to input HSI file (.npz)"),
    output: str = typer.Argument(..., help="Path to output upscaled HSI file (.npz)"),
    scale: int = typer.Option(2, "--scale", "-s", help="Upscaling factor"),
    method: str = typer.Option(
        "bicubic",
        "--method", "-m",
        help="Upscaling method: bicubic, bilinear, nearest, edge_guided"
    ),
    guide: Optional[str] = typer.Option(
        None,
        "--guide", "-g",
        help="Path to RGB guide image (required for edge_guided)"
    ),
):
    """Upscale an HSI cube."""
    pass

@app.command()
def evaluate(
    input: str = typer.Argument(..., help="Path to input HSI file (.npz)"),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Path to output metrics JSON file"
    ),
    roi: Optional[str] = typer.Option(
        None,
        "--roi", "-r",
        help="Path to ROI mask file (.npy)"
    ),
):
    """Evaluate HSI quality metrics."""
    pass

@app.command()
def info(
    input: str = typer.Argument(..., help="Path to HSI file (.npz) to inspect"),
):
    """Display information about an HSI cube file."""
    pass

# TODO Implement variety of annotation files with different formats
@app.command()
def crop(
    input_hsi: str = typer.Argument(..., help="Path to input HSI file (.npz)"),
    output_dir: str = typer.Argument(..., help="Directory to save cropped regions"),
    annotations: str = typer.Option(
        ...,
        "--annotations", "-a",
        help="Path to annotations file"
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels", "-l",
        help="Comma-separated list of labels to include"
    ),
    min_size: int = typer.Option(
        16,
        "--min-size",
        help="Minimum crop size in pixels"
    ),
    square: bool = typer.Option(
        True,
        "--square/--no-square",
        help="Make crops square"
    ),
):
    """Extract ROI crops from an HSI cube."""
    pass

def main():
    app()

if __name__ == "__main":
    main()