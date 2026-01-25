"""Unit tests for dataset runner."""

from hsi_pipeline.dataset.runner import (
    run_dataset, write_dataset_report, OnErrorPolicy, DatasetReport, SampleResult
)
from hsi_pipeline.manifest.parser import Manifest, Sample


class TestRunDataset:
    """Tests for run_dataset function."""

    def test_processes_all_samples(self, tmp_path):
        """Should process all samples in manifest."""
        manifest = Manifest(
            root=tmp_path,
            samples=[
                Sample(id="s1", image="img1.png", image_resolved=tmp_path / "img1.png"),
                Sample(id="s2", image="img2.png", image_resolved=tmp_path / "img2.png"),
            ],
            source_path=tmp_path / "manifest.yaml"
        )
        
        processed = []
        def process_fn(sample, out):
            processed.append(sample.id)
        
        report = run_dataset(manifest, tmp_path / "out", process_fn)
        
        assert len(processed) == 2
        assert "s1" in processed
        assert "s2" in processed
        assert report.processed_ok == 2
        assert report.failed == 0

    def test_continue_on_error(self, tmp_path):
        """Should continue processing after failure with CONTINUE policy."""
        manifest = Manifest(
            root=tmp_path,
            samples=[
                Sample(id="s1", image="img1.png", image_resolved=tmp_path / "img1.png"),
                Sample(id="s2", image="img2.png", image_resolved=tmp_path / "img2.png"),
                Sample(id="s3", image="img3.png", image_resolved=tmp_path / "img3.png"),
            ],
            source_path=tmp_path / "manifest.yaml"
        )
        
        def process_fn(sample, out):
            if sample.id == "s2":
                raise ValueError("Test error")
        
        report = run_dataset(manifest, tmp_path / "out", process_fn, on_error=OnErrorPolicy.CONTINUE)
        
        assert report.processed_ok == 2
        assert report.failed == 1
        assert len(report.failures) == 1
        assert report.failures[0]["sample_id"] == "s2"

    def test_abort_on_error(self, tmp_path):
        """Should stop processing after failure with ABORT policy."""
        manifest = Manifest(
            root=tmp_path,
            samples=[
                Sample(id="s1", image="img1.png", image_resolved=tmp_path / "img1.png"),
                Sample(id="s2", image="img2.png", image_resolved=tmp_path / "img2.png"),
                Sample(id="s3", image="img3.png", image_resolved=tmp_path / "img3.png"),
            ],
            source_path=tmp_path / "manifest.yaml"
        )
        
        processed = []
        def process_fn(sample, out):
            processed.append(sample.id)
            if sample.id == "s2":
                raise ValueError("Test error")
        
        report = run_dataset(manifest, tmp_path / "out", process_fn, on_error=OnErrorPolicy.ABORT)
        
        assert "s1" in processed
        assert "s2" in processed
        assert "s3" not in processed
        assert report.failed == 1

    def test_creates_sample_directories(self, tmp_path):
        """Should create output directory per sample."""
        manifest = Manifest(
            root=tmp_path,
            samples=[
                Sample(id="s1", image="img1.png", image_resolved=tmp_path / "img1.png"),
            ],
            source_path=tmp_path / "manifest.yaml"
        )
        
        def process_fn(sample, out):
            assert out.exists()
            assert out.name == "s1"
        
        out_dir = tmp_path / "output"
        run_dataset(manifest, out_dir, process_fn)
        
        assert (out_dir / "s1").exists()

    def test_records_execution_time(self, tmp_path):
        """Should record execution time for each sample."""
        manifest = Manifest(
            root=tmp_path,
            samples=[
                Sample(id="s1", image="img1.png", image_resolved=tmp_path / "img1.png"),
            ],
            source_path=tmp_path / "manifest.yaml"
        )
        
        def process_fn(sample, out):
            import time
            time.sleep(0.1)
        
        report = run_dataset(manifest, tmp_path / "out", process_fn)
        
        assert report.results[0].execution_time >= 0.1
        assert report.total_time >= 0.1


class TestWriteDatasetReport:
    """Tests for write_dataset_report function."""

    def test_writes_report_json(self, tmp_path):
        """Should write dataset_report.json."""
        report = DatasetReport(
            total_samples=3,
            processed_ok=2,
            failed=1,
            failures=[{"sample_id": "s2", "reason": "Test error"}],
            results=[
                SampleResult(sample_id="s1", success=True, output_dir=tmp_path / "s1"),
                SampleResult(sample_id="s2", success=False, error="Test error"),
                SampleResult(sample_id="s3", success=True, output_dir=tmp_path / "s3"),
            ],
            total_time=1.5
        )
        
        report_path = write_dataset_report(report, tmp_path)
        
        assert report_path.exists()
        assert report_path.name == "dataset_report.json"
        
        import json
        with open(report_path) as f:
            data = json.load(f)
        
        assert data["total_samples"] == 3
        assert data["processed_ok"] == 2
        assert data["failed"] == 1
        assert len(data["failures"]) == 1
        assert len(data["samples"]) == 3
