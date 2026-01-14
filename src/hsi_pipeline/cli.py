"""Command-line interface for SpectraPipe - HSI Pipeline."""

from pathlib import Path
from typing import Optional
from .pipeline.run import load_config
from .transforms.rgb_to_hsi import rgb_to_hsi
from .preprocess.input_fitting import fit_input, unfit_output
from .export.manager import ExportManager
import typer
from rich.console import Console

PIPELINE_VERSION = "0.1.0"

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
    roi_mask: Optional[Path] = typer.Option(
        None,
        "--roi-mask", "-r",
        help="Path to ROI mask image for separability calculation",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True
    ),
):
    """Run the full HSI processing pipeline."""
    import cv2
    from PIL import Image, UnidentifiedImageError 

    import time
    from rich.table import Table

    start_time = time.perf_counter()

    cfg = load_config(config)

    if out is None:
        out = input.parent / "output"

    exporter = ExportManager(out_dir=out, format="npz", overwrite=True)

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

    console.print(f"Fitting input: policy={cfg.fitting.policy}, multiple={cfg.fitting.multiple}")
    try:
        fit_result = fit_input(rgb, multiple=cfg.fitting.multiple, policy=cfg.fitting.policy)
    except ValueError as e:
        console.print(f"[red]Fitting Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure image has valid dimensions and is RGB (H, W, 3).")
        raise typer.Exit(1)

    console.print(f"  Original shape: {fit_result.original_shape[0]}x{fit_result.original_shape[1]}")
    console.print(f"  Fitted shape:   {fit_result.fitted_shape[0]}x{fit_result.fitted_shape[1]}")

    console.print("Converting RGB → HSI using MST++...")

    use_ensemble = not no_ensemble
    
    try:
        exporter.prepare_directory()

        hsi_fitted = rgb_to_hsi(fit_result.fitted, cfg, ensemble_override=use_ensemble)
        
        hsi = unfit_output(hsi_fitted, fit_result.original_shape, fit_result.padding)

        exporter.export_array("hsi_raw", hsi)

        # ROI processing
        roi_result = None
        raw_separability = None
        
        if roi_mask is not None:
            from .roi.loader import load_roi_mask, ROILoadError, ROIValidationError
            from .roi.separability import calculate_separability
            
            console.print(f"Loading ROI mask: {roi_mask}")
            try:
                roi_result = load_roi_mask(roi_mask, fit_result.original_shape[:2])
                
                for warning in roi_result.warnings:
                    console.print(f"[yellow]ROI Warning:[/yellow] {warning}")
                
                console.print(f"  ROI coverage: {roi_result.coverage:.1%}")
                
                console.print("Calculating separability...")
                raw_separability = calculate_separability(hsi, roi_result.mask)
                
                if raw_separability is not None:
                    console.print(f"  Raw separability: {raw_separability:.4f}")
                else:
                    console.print("  Raw separability: [yellow]NA[/yellow] (empty or full ROI)")
                    
            except ROILoadError as e:
                console.print(f"[red]ROI Load Error:[/red] {e}")
                console.print("[yellow]Suggestion:[/yellow] Check that the mask file exists and is a valid image.")
                raise typer.Exit(1)
            except ROIValidationError as e:
                console.print(f"[red]ROI Validation Error:[/red] {e}")
                console.print("[yellow]Suggestion:[/yellow] Ensure mask dimensions match input image.")
                raise typer.Exit(1)
        else:
            console.print("[dim]ROI not provided → separability omitted[/dim]")

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Export metrics with ROI data
        roi_extra = {}
        if roi_result is not None:
            roi_extra["raw_separability"] = raw_separability
            roi_extra["roi_coverage"] = roi_result.coverage
            roi_extra["roi_mask_path"] = roi_result.path
        
        exporter.export_metrics(
            hsi_shape=hsi.shape,
            execution_time=duration,
            ensemble_enabled=use_ensemble,
            extra=roi_extra if roi_extra else None,
        )

        fitting_info = {
            "policy": fit_result.policy,
            "multiple": cfg.fitting.multiple,
            "padding": list(fit_result.padding),
            "input_shape_original": list(fit_result.original_shape),
            "input_shape_fitted": list(fit_result.fitted_shape),
        }
        
        run_config_extra = {
            "ensemble_enabled": use_ensemble,
            "hsi_shape": list(hsi.shape)
        }
        
        if roi_result is not None:
            run_config_extra["roi"] = {
                "mask_path": roi_result.path,
                "policy": "fail_on_invalid",
                "binarize_threshold": 127,
                "coverage": roi_result.coverage,
            }
        
        exporter.export_run_config(
            input_path=str(input),
            config_path=str(config),
            fitting_info=fitting_info,
            pipeline_version=PIPELINE_VERSION,
            extra=run_config_extra,
        )

    except NotADirectoryError as e:
        console.print(f"[red]Output Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Ensure --out points to a directory, not a file.")
        raise typer.Exit(1)
    except PermissionError as e:
        console.print(f"[red]Permission Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Check folder permissions or choose a different output path.")
        raise typer.Exit(1)
    except FileExistsError as e:
        console.print(f"[red]Collision Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Use a different output directory or enable overwrite.")
        raise typer.Exit(1)
    except OSError as e:
        console.print("[red]System Error:[/red] Failed to create/access output directory.")
        console.print(f"Detail: {e}")
        raise typer.Exit(1)
    except Exception as e:
        removed = exporter.cleanup_partial()
        console.print("[red]Pipeline Error:[/red] Execution failed unexpectedly.")
        console.print(f"[yellow]Detail:[/yellow] {e}")
        if removed:
            console.print(f"[yellow]Cleanup:[/yellow] Removed {len(removed)} partial artifact(s): {', '.join(removed)}")
        console.print("[yellow]Suggestion:[/yellow] Check input image and model configuration.")
        raise typer.Exit(1)

    exported = exporter.list_exported()
    console.print(f"\n[bold]Exported artifacts ({len(exported)}):[/bold]")
    for artifact in exported:
        console.print(f"  • {artifact}")

    table = Table(title="SpectraPipe Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input Image", str(input))
    table.add_row("Original Shape", f"{fit_result.original_shape[0]}x{fit_result.original_shape[1]}")
    table.add_row("Fitted Shape", f"{fit_result.fitted_shape[0]}x{fit_result.fitted_shape[1]}")
    table.add_row("Fitting Policy", fit_result.policy)
    table.add_row("Output Directory", str(out))
    table.add_row("Config File", str(config))
    table.add_row("Ensemble Enabled", "No" if no_ensemble else "Yes")
    table.add_row("HSI Shape", str(hsi.shape))
    table.add_row("Total Time", f"{duration:.2f}s")

    console.print(table)
    console.print("[green]✓[/green] Pipeline finished successfully.")


@app.command()
def metrics(
    from_dir: Path = typer.Option(
        ...,
        "--from", "-f",
        help="Output directory containing metrics.json",
        resolve_path=True
    ),
):
    """Display a human-readable summary of metrics from a previous run."""
    from .metrics.reader import read_metrics, MetricsNotFoundError, MetricsCorruptError
    from .metrics.formatter import format_metrics, print_warnings
    
    # Validate directory
    if not from_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Check the path and try again.")
        raise typer.Exit(1)
    
    if not from_dir.is_dir():
        console.print(f"[red]Error:[/red] Path is not a directory: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Provide a directory path, not a file.")
        raise typer.Exit(1)
    
    metrics_path = from_dir / "metrics.json"
    
    try:
        result = read_metrics(metrics_path)
    except MetricsNotFoundError:
        console.print(f"[red]Error:[/red] metrics.json not found in: {from_dir}")
        console.print("[yellow]Suggestion:[/yellow] Run `spectrapipe run --input <image> --out <dir>` first to generate metrics.")
        raise typer.Exit(1)
    except MetricsCorruptError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Suggestion:[/yellow] Regenerate metrics with `spectrapipe run --input <image> --out <dir>`.")
        raise typer.Exit(1)
    
    # Print warnings if any
    if result.warnings:
        print_warnings(result.warnings, console)
    
    # Print formatted metrics
    format_metrics(result.data, console)
    
    console.print("[green]✓[/green] Metrics loaded successfully.")


def main():
    app()

if __name__ == "__main":
    main()