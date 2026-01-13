"""Metrics formatter for human-readable output."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def format_metrics(data: dict, console: Console) -> None:
    """Print formatted metrics to console.
    
    Args:
        data: Metrics dictionary.
        console: Rich console for output.
    """
    console.print()
    console.print(Panel.fit(
        "[bold magenta]SpectraPipe Metrics Summary[/bold magenta]",
        border_style="magenta"
    ))
    console.print()
    
    # General Stats
    _print_general_stats(data, console)
    
    # Separability (if exists)
    _print_separability(data, console)
    
    # Clean Metrics (if exists)
    _print_clean_metrics(data, console)
    
    # Upscaling (if exists)
    _print_upscaling(data, console)


def _print_general_stats(data: dict, console: Console) -> None:
    """Print general statistics section."""
    table = Table(title="📊 General Stats", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # HSI Shape
    hsi_shape = data.get("hsi_shape")
    if hsi_shape:
        table.add_row("HSI Shape", str(tuple(hsi_shape)))
    
    # Bands
    n_bands = data.get("n_bands")
    if n_bands is not None:
        table.add_row("Bands", str(n_bands))
    
    # Execution time
    exec_time = data.get("execution_time_seconds")
    if exec_time is not None:
        table.add_row("Execution Time", f"{exec_time:.2f}s")
    
    # Ensemble
    ensemble = data.get("ensemble_enabled")
    if ensemble is not None:
        table.add_row("Ensemble", "Yes" if ensemble else "No")
    
    # Timestamp
    timestamp = data.get("timestamp")
    if timestamp:
        table.add_row("Timestamp", timestamp)
    
    if table.row_count > 0:
        console.print(table)
        console.print()


def _print_separability(data: dict, console: Console) -> None:
    """Print separability metrics if available."""
    raw_sep = data.get("raw_separability")
    
    if raw_sep is None:
        return
    
    table = Table(title="📐 Separability", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    if isinstance(raw_sep, (int, float)):
        table.add_row("Raw Separability", f"{raw_sep:.4f}")
    else:
        table.add_row("Raw Separability", str(raw_sep))
    
    console.print(table)
    console.print()


def _print_clean_metrics(data: dict, console: Console) -> None:
    """Print clean metrics if available."""
    clean_sep = data.get("clean_separability")
    sam = data.get("raw_clean_sam")
    rmse = data.get("raw_clean_rmse")
    
    if clean_sep is None and sam is None and rmse is None:
        return
    
    table = Table(title="🧹 Clean Metrics", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    if clean_sep is not None:
        val = f"{clean_sep:.4f}" if isinstance(clean_sep, (int, float)) else str(clean_sep)
        table.add_row("Clean Separability", val)
    
    if sam is not None:
        val = f"{sam:.4f}" if isinstance(sam, (int, float)) else str(sam)
        table.add_row("SAM (Raw→Clean)", val)
    
    if rmse is not None:
        val = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
        table.add_row("RMSE (Raw→Clean)", val)
    
    console.print(table)
    console.print()


def _print_upscaling(data: dict, console: Console) -> None:
    """Print upscaling metadata if available."""
    upscaled_size = data.get("upscaled_size")
    upscale_factor = data.get("upscale_factor")
    
    if upscaled_size is None and upscale_factor is None:
        return
    
    table = Table(title="📈 Upscaling", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    if upscale_factor is not None:
        table.add_row("Factor", f"{upscale_factor}x")
    
    if upscaled_size is not None:
        if isinstance(upscaled_size, list):
            table.add_row("Upscaled Size", str(tuple(upscaled_size)))
        else:
            table.add_row("Upscaled Size", str(upscaled_size))
    
    console.print(table)
    console.print()


def print_warnings(warnings: list[str], console: Console) -> None:
    """Print warnings for missing/invalid fields."""
    if not warnings:
        return
    
    console.print("[yellow]⚠ Warnings:[/yellow]")
    for warning in warnings:
        console.print(f"  • {warning}")
    console.print()
