"""Wrapper for MST++ spectral reconstruction model."""

import sys
from pathlib import Path
from typing import Literal
import itertools

import numpy as np
import torch

_MST_PATH = Path(__file__).parent.parent.parent.parent / "external" / "MST-plus-plus" / "predict_code"
if str(_MST_PATH) not in sys.path:
    sys.path.insert(0, str(_MST_PATH))

from architecture import model_generator

class MSTppWrapper:
    """Wrapper for MST++ spectral reconstruction model.
    
    This class provides a clean interface for using the MST++ model
    for RGB to hyperspectral image reconstruction.
    
    Attributes:
        device: The device (cuda/cpu) to run inference on.
        model: The loaded MST++ model.
    """

    def __init__(self, weights_path: str | Path, device: str = "auto"):
        """Initialize the MST++ wrapper.
        
        Args:
            weights_path: Path to the pretrained model weights (.pth file).
            device: Device to use - "auto", "cuda", or "cpu".
        """
        self.weights_path = Path(weights_path)
        self.device = self._get_device(device)
        self.model = self._load_model()

    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self) -> torch.nn.Module:
        """Load the MST++ model with pretrained weights."""
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. "
                "Please download from: https://drive.google.com/file/d/18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM"
            )
        
        import contextlib
        import os
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
             model = model_generator("mst_plus_plus", str(self.weights_path))
        
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self, 
        rgb: np.ndarray, 
        ensemble: bool = True, 
        ensemble_mode: Literal["mean", "median"] = "mean"
    ) -> np.ndarray:
        """Convert RGB image to 31-band HSI using MST++.
        
        Args:
            rgb: (H, W, 3) RGB image, uint8 [0-255] or float32 [0-1].
            ensemble: Whether to use test-time augmentation (8x transforms).
            ensemble_mode: How to aggregate ensemble predictions - 'mean' or 'median'.
        
        Returns:
            hsi: (H, W, 31) hyperspectral cube, float32 [0, 1].
        """
        rgb_processed = self._preprocess(rgb)
        
        x = torch.from_numpy(rgb_processed.transpose(2, 0, 1)).unsqueeze(0).float()
        x = x.to(self.device)
        
        with torch.no_grad():
            if ensemble:
                result = self._forward_ensemble(x, ensemble_mode)
            else:
                result = self.model(x)
        
        hsi = result.cpu().numpy().squeeze().transpose(1, 2, 0)
        return np.clip(hsi, 0, 1).astype(np.float32)
    
    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Preprocess RGB image for model input."""

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        
        rgb_min, rgb_max = rgb.min(), rgb.max()
        if rgb_max - rgb_min > 1e-8:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        
        return rgb.astype(np.float32)
    
    def _forward_ensemble(
        self, 
        x: torch.Tensor, 
        mode: Literal["mean", "median"]
    ) -> torch.Tensor:
        """Test-time augmentation with 8 geometric transforms.
        
        Applies all combinations of horizontal flip, vertical flip,
        and transpose, then aggregates the results.
        """
        def _transform(data: torch.Tensor, xflip: bool, yflip: bool, 
                       transpose: bool, reverse: bool = False) -> torch.Tensor:
            if not reverse:
                if xflip:
                    data = torch.flip(data, [3])
                if yflip:
                    data = torch.flip(data, [2])
                if transpose:
                    data = torch.transpose(data, 2, 3)
            else:
                if transpose:
                    data = torch.transpose(data, 2, 3)
                if yflip:
                    data = torch.flip(data, [2])
                if xflip:
                    data = torch.flip(data, [3])
            return data
        
        outputs = []
        for xflip, yflip, transpose in itertools.product([False, True], repeat=3):
            data = x.clone()
            data = _transform(data, xflip, yflip, transpose)
            data = self.model(data)
            outputs.append(_transform(data, xflip, yflip, transpose, reverse=True))
        
        stacked = torch.stack(outputs, dim=0)
        if mode == "mean":
            return stacked.mean(dim=0)
        else:
            return stacked.median(dim=0)[0]
    
    def __repr__(self) -> str:
        return f"MSTppWrapper(weights='{self.weights_path}', device='{self.device}')"