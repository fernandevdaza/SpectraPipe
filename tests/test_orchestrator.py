"""Unit tests for PipelineOrchestrator without CLI simulation."""

from unittest.mock import patch
import numpy as np
import pytest

from hsi_pipeline.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineInput,
    PipelineOutput,
)
from hsi_pipeline.pipeline.run import load_config
from hsi_pipeline.types import RunConfig


class TestPipelineOrchestrator:
    """Unit tests for PipelineOrchestrator."""
    
    @patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
    def test_minimal_run(self, mock_rgb_to_hsi):
        """Test orchestrator produces valid output with minimal input."""
        mock_rgb_to_hsi.return_value = np.zeros((64, 64, 31), dtype=np.float32)
        
        config = load_config("configs/defaults.yaml")
        rgb = np.zeros((60, 60, 3), dtype=np.uint8)  # Odd size to test fitting
        
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(PipelineInput(
            rgb=rgb,
            config=config,
            roi_mask_path=None,
            upscale_factor=None,
            use_ensemble=False,
        ))
        
        assert isinstance(result, PipelineOutput)
        assert result.hsi_raw.shape == (60, 60, 31)  # Unfitted to original
        assert result.fit_result is not None
        assert result.execution_time > 0
        assert result.roi_data is None
        assert result.clean_data is None
        assert result.upscale_data is None
    
    @patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
    def test_with_upscaling(self, mock_rgb_to_hsi):
        """Test orchestrator applies upscaling when factor is provided."""
        mock_rgb_to_hsi.return_value = np.random.rand(64, 64, 31).astype(np.float32)
        
        config = load_config("configs/defaults.yaml")
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(PipelineInput(
            rgb=rgb,
            config=config,
            upscale_factor=2,
        ))
        
        assert result.upscale_data is not None
        assert result.upscale_data.factor == 2
        assert result.upscale_data.hsi_baseline.shape == (128, 128, 31)
        assert result.upscale_data.hsi_improved.shape == (128, 128, 31)
    
    @patch("hsi_pipeline.pipeline.orchestrator.rgb_to_hsi", autospec=True)
    def test_with_roi_mask(self, mock_rgb_to_hsi, tmp_path):
        """Test orchestrator processes ROI mask correctly."""
        mock_rgb_to_hsi.return_value = np.random.rand(64, 64, 31).astype(np.float32)
        
        # Create a simple ROI mask
        from PIL import Image
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = 255  # Top half is ROI
        mask_path = tmp_path / "roi_mask.png"
        Image.fromarray(mask).save(mask_path)
        
        config = load_config("configs/defaults.yaml")
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(PipelineInput(
            rgb=rgb,
            config=config,
            roi_mask_path=mask_path,
        ))
        
        assert result.roi_data is not None
        assert result.roi_data.coverage == pytest.approx(0.5, rel=0.1)
        assert result.raw_separability is not None
        assert result.hsi_clean is not None  # Partial coverage triggers clean
        assert result.clean_data is not None
    
    def test_fit_result_contains_metadata(self):
        """Test that fit_result contains expected metadata."""
        from hsi_pipeline.preprocess.input_fitting import FittingResult
        
        config = load_config("configs/defaults.yaml")
        rgb = np.zeros((65, 97, 3), dtype=np.uint8)  # Odd dimensions
        
        orchestrator = PipelineOrchestrator()
        fit_result = orchestrator._fit_input(rgb, config)
        
        assert isinstance(fit_result, FittingResult)
        assert fit_result.original_shape == (65, 97)
        assert fit_result.fitted_shape[0] % 32 == 0
        assert fit_result.fitted_shape[1] % 32 == 0
        assert fit_result.policy == "pad_to_multiple"


class TestPipelineInputOutput:
    """Test dataclasses for pipeline I/O."""
    
    def test_pipeline_input_defaults(self):
        """Test PipelineInput has correct defaults."""
        config = RunConfig()
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        
        inp = PipelineInput(rgb=rgb, config=config)
        
        assert inp.use_ensemble is True
        assert inp.roi_mask_path is None
        assert inp.upscale_factor is None
    
    def test_pipeline_output_optional_fields(self):
        """Test PipelineOutput optional fields are None by default."""
        from hsi_pipeline.preprocess.input_fitting import FittingResult
        
        hsi = np.zeros((32, 32, 31), dtype=np.float32)
        fit = FittingResult(
            fitted=np.zeros((32, 32, 3)),
            original_shape=(32, 32),
            fitted_shape=(32, 32),
            policy="pad_to_multiple",
            padding=(0, 0, 0, 0),
        )
        
        output = PipelineOutput(hsi_raw=hsi, fit_result=fit)
        
        assert output.hsi_clean is None
        assert output.roi_data is None
        assert output.clean_data is None
        assert output.upscale_data is None
        assert output.raw_separability is None
        assert output.clean_metrics is None
