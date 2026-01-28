"""Tests for US-19 Reproducibility."""

import json
import pytest
import numpy as np
from PIL import Image
from typer.testing import CliRunner
from unittest.mock import patch

from hsi_pipeline.cli import app

runner = CliRunner()


class TestReproducibility:
    """End-to-end reproducibility tests (config propagation)."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with images."""
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        Image.new("RGB", (100, 100), color="white").save(img1)
        Image.new("RGB", (100, 100), color="black").save(img2)
        return tmp_path, img1, img2

    @patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
    def test_run_config_reproducibility(self, mock_rgb_to_hsi, workspace):
        """Run 1 config should define Run 2 parameters."""
        tmp_path, img1, img2 = workspace

        # Make inference deterministic and fast (avoid MST++ / weights / architecture import)
        mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)

        # Run 1: Custom parameters
        run1_out = tmp_path / "run1"
        result1 = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(img1),
                "--out",
                str(run1_out),
                "--upscale-factor",
                "2",
                "--no-ensemble",
            ],
        )
        assert result1.exit_code == 0, (
            f"STDOUT/ERR:\n{result1.stdout if hasattr(result1, 'stdout') else result1.output}\n"
            f"EXC:\n{repr(result1.exception)}"
        )

        # Verify Run 1 config
        config1_path = run1_out / "run_config.json"
        assert config1_path.exists()

        with open(config1_path) as f:
            data1 = json.load(f)

        assert "config" in data1
        assert "meta" in data1
        assert data1["config"]["upscaling"]["enabled"] is True
        assert data1["config"]["upscaling"]["factor"] == 2
        assert data1["config"]["model"]["ensemble"] is False

        # Run 2: Use run_config.json from Run 1
        run2_out = tmp_path / "run2"
        result2 = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(img2),
                "--out",
                str(run2_out),
                "--config",
                str(config1_path),
            ],
        )
        assert result2.exit_code == 0, (
            f"STDOUT/ERR:\n{result2.stdout if hasattr(result2, 'stdout') else result2.output}\n"
            f"EXC:\n{repr(result2.exception)}"
        )

        # Verify Run 2 adhered to config
        config2_path = run2_out / "run_config.json"
        with open(config2_path) as f:
            data2 = json.load(f)

        assert data2["config"]["upscaling"]["enabled"] is True
        assert data2["config"]["upscaling"]["factor"] == 2
        assert data2["config"]["model"]["ensemble"] is False

        # Metadata should be new
        assert data1["meta"]["timestamp"] != data2["meta"]["timestamp"]
        assert data2["meta"]["input_path"] == str(img2)

    @patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
    def test_dataset_reproducibility(self, mock_rgb_to_hsi, workspace):
        """Dataset command should also accept run_config.json."""
        tmp_path, img1, _ = workspace

        mock_rgb_to_hsi.return_value = np.zeros((32, 32, 31), dtype=np.float32)

        # Create manifest
        manifest_path = tmp_path / "manifest.yaml"
        manifest_content = f"""
root: {tmp_path}
samples:
  - id: s1
    image: img1.png
"""
        manifest_path.write_text(manifest_content)

        # Create a config manually representing a previous run
        config_json_path = tmp_path / "manual_config.json"
        manual_config = {
            "meta": {"irrelevant": "metadata"},
            "config": {
                "upscaling": {"enabled": True, "factor": 4},
                "fitting": {"multiple": 16},
            },
        }
        with open(config_json_path, "w") as f:
            json.dump(manual_config, f)

        # Run dataset with JSON config
        out_dir = tmp_path / "dataset_out"
        result = runner.invoke(
            app,
            [
                "dataset",
                "--manifest",
                str(manifest_path),
                "--out",
                str(out_dir),
                "--config",
                str(config_json_path),
            ],
        )
        assert result.exit_code == 0, (
            f"STDOUT/ERR:\n{result.stdout if hasattr(result, 'stdout') else result.output}\n"
            f"EXC:\n{repr(result.exception)}"
        )

        # Verify output config of sample
        sample_config_path = out_dir / "s1" / "run_config.json"
        assert sample_config_path.exists()

        with open(sample_config_path) as f:
            s_data = json.load(f)

        assert s_data["config"]["upscaling"]["factor"] == 4
        assert s_data["config"]["fitting"]["multiple"] == 16
