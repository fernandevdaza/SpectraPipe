"""Dataset runner for batch processing."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from enum import Enum

from ..manifest.parser import Manifest, Sample


class OnErrorPolicy(str, Enum):
    """Policy for handling sample processing errors."""
    CONTINUE = "continue"
    ABORT = "abort"


@dataclass
class SampleResult:
    """Result of processing a single sample."""
    sample_id: str
    success: bool
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class DatasetReport:
    """Aggregate report for dataset processing."""
    total_samples: int
    processed_ok: int
    failed: int
    failures: List[dict] = field(default_factory=list)
    results: List[SampleResult] = field(default_factory=list)
    total_time: float = 0.0


def run_dataset(
    manifest: Manifest,
    output_dir: Path,
    process_fn: Callable[[Sample, Path], None],
    on_error: OnErrorPolicy = OnErrorPolicy.CONTINUE,
    console: Optional[Any] = None,
) -> DatasetReport:
    """Run pipeline on all samples in a manifest.
    
    Args:
        manifest: Parsed manifest with samples
        output_dir: Base output directory
        process_fn: Function to process each sample (sample, output_path) -> None
        on_error: Policy for handling errors
        console: Optional rich console for logging
    
    Returns:
        DatasetReport with aggregate results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[SampleResult] = []
    start_time = time.perf_counter()
    
    for i, sample in enumerate(manifest.samples):
        sample_out = output_dir / sample.id
        sample_out.mkdir(parents=True, exist_ok=True)
        
        if console:
            console.print(
                f"[{i+1}/{len(manifest.samples)}] Processing sample: {sample.id}"
            )
        
        sample_start = time.perf_counter()
        
        try:
            process_fn(sample, sample_out)
            sample_time = time.perf_counter() - sample_start
            
            results.append(SampleResult(
                sample_id=sample.id,
                success=True,
                output_dir=sample_out,
                execution_time=sample_time
            ))
            
            if console:
                console.print(f"  ✓ Completed in {sample_time:.2f}s")
                
        except Exception as e:
            sample_time = time.perf_counter() - sample_start
            error_msg = str(e)
            
            results.append(SampleResult(
                sample_id=sample.id,
                success=False,
                output_dir=sample_out,
                error=error_msg,
                execution_time=sample_time
            ))
            
            if console:
                console.print(f"  [red]✗ Failed:[/red] {error_msg}")
            
            if on_error == OnErrorPolicy.ABORT:
                if console:
                    console.print("[red]Aborting due to --on-error abort[/red]")
                break
    
    total_time = time.perf_counter() - start_time
    
    processed_ok = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    failures = [
        {"sample_id": r.sample_id, "reason": r.error}
        for r in results if not r.success
    ]
    
    return DatasetReport(
        total_samples=len(manifest.samples),
        processed_ok=processed_ok,
        failed=failed,
        failures=failures,
        results=results,
        total_time=total_time
    )


def write_dataset_report(report: DatasetReport, output_dir: Path) -> Path:
    """Write dataset report to JSON file.
    
    Args:
        report: DatasetReport to write
        output_dir: Directory to write report to
    
    Returns:
        Path to written report file
    """
    report_path = output_dir / "dataset_report.json"
    
    report_data = {
        "total_samples": report.total_samples,
        "processed_ok": report.processed_ok,
        "failed": report.failed,
        "total_time_seconds": round(report.total_time, 2),
        "failures": report.failures,
        "samples": [
            {
                "sample_id": r.sample_id,
                "success": r.success,
                "output_dir": str(r.output_dir) if r.output_dir else None,
                "error": r.error,
                "execution_time_seconds": round(r.execution_time, 2)
            }
            for r in report.results
        ]
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    return report_path
