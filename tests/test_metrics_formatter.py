"""Unit tests for metrics formatter."""

from io import StringIO
from rich.console import Console
from hsi_pipeline.metrics.formatter import (
    format_metrics,
    print_warnings,
)


class TestFormatMetrics:
    """Tests for format_metrics function."""

    def test_complete_data(self):
        """Should print all sections for complete data."""
        data = {
            "hsi_shape": [512, 512, 31],
            "n_bands": 31,
            "execution_time_seconds": 12.5,
            "ensemble_enabled": True,
            "timestamp": "2026-01-13T12:00:00"
        }
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        format_metrics(data, console)
        
        result = output.getvalue()
        assert "General Stats" in result
        assert "31" in result
        assert "12.5" in result or "12.50" in result

    def test_partial_data_no_crash(self):
        """Should handle partial data without crashing."""
        data = {"hsi_shape": [100, 100, 31]}
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        # Should not raise
        format_metrics(data, console)
        
        result = output.getvalue()
        assert "General Stats" in result

    def test_separability_section(self):
        """Should print separability if available."""
        data = {
            "hsi_shape": [100, 100, 31],
            "raw_separability": 0.85
        }
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        format_metrics(data, console)
        
        result = output.getvalue()
        assert "Separability" in result
        assert "0.85" in result

    def test_clean_metrics_section(self):
        """Should print clean metrics if available."""
        data = {
            "hsi_shape": [100, 100, 31],
            "clean_separability": 0.92,
            "raw_clean_sam": 0.05,
            "raw_clean_rmse": 0.02
        }
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        format_metrics(data, console)
        
        result = output.getvalue()
        assert "Clean Metrics" in result

    def test_upscaling_section(self):
        """Should print upscaling if available."""
        data = {
            "hsi_shape": [100, 100, 31],
            "upscale_factor": 2,
            "upscaled_size": [200, 200]
        }
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        format_metrics(data, console)
        
        result = output.getvalue()
        assert "Upscaling" in result
        assert "2x" in result

    def test_empty_data_no_crash(self):
        """Should handle empty data without crashing."""
        data = {}
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        # Should not raise
        format_metrics(data, console)


class TestPrintWarnings:
    """Tests for print_warnings function."""

    def test_prints_warnings(self):
        """Should print all warnings."""
        warnings = ["Missing field: n_bands", "Missing field: timestamp"]
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        print_warnings(warnings, console)
        
        result = output.getvalue()
        assert "n_bands" in result
        assert "timestamp" in result

    def test_no_output_for_empty_warnings(self):
        """Should not print anything for empty warnings."""
        warnings = []
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        print_warnings(warnings, console)
        
        result = output.getvalue()
        assert result == ""
