import numpy as np

from typing import TYPE_CHECKING
from ..models.mst_wrapper import MSTppWrapper

if TYPE_CHECKING:
    from ..types import RunConfig

_model: "MSTppWrapper | None" = None


def get_model(config: "RunConfig") -> "MSTppWrapper":
    """Get or create MST++ model instance (singleton).
    
    The model is loaded once and reused for all subsequent calls.
    
    Args:
        config: Pipeline configuration containing model settings.
    
    Returns:
        MSTppWrapper instance.
    """
    global _model
    
    if _model is None:
        _model = MSTppWrapper(
            weights_path=config.model.weights_path,
            device=config.model.device
        )
    
    return _model


def reset_model() -> None:
    """Reset the global model instance.
    
    Useful for testing or when switching configurations.
    """
    global _model
    _model = None

def rgb_to_hsi(
    rgb: np.ndarray, 
    config: "RunConfig", 
    ensemble_override: bool | None = None
) -> np.ndarray:
    """Convert RGB image to 31-band hyperspectral cube using MST++.
    
    Args:
        rgb: (H, W, 3) RGB image, uint8 [0-255] or float32 [0-1].
        config: Pipeline configuration.
        ensemble_override: If provided, overrides the ensemble setting.
    
    Returns:
        hsi: (H, W, 31) hyperspectral cube, float32 [0, 1].
    """
    model = get_model(config)
    
    use_ensemble = True
    if ensemble_override is not None:
        use_ensemble = ensemble_override

    return model.predict(
        rgb,
        ensemble=use_ensemble,
        ensemble_mode=config.model.ensemble_mode
    )

    