"""Tests for configuration loading and management."""

import warnings

import pytest

from hsi_pipeline.pipeline.run import (
    load_config, 
    merge_cli_overrides, 
    DEFAULT_CONFIG_TEMPLATE,
)
from hsi_pipeline.types import RunConfig


class TestConfigLoading:
    """Tests for config loading functionality."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(DEFAULT_CONFIG_TEMPLATE)
        
        cfg = load_config(config_path)
        
        assert isinstance(cfg, RunConfig)
        assert cfg.model.name == "mst_plus_plus"
        assert cfg.model.ensemble is True
        assert cfg.fitting.multiple == 32
    
    def test_load_partial_config(self, tmp_path):
        """Test loading config with only some fields (rest use defaults)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
model:
  device: cpu
  ensemble: false
""")
        
        cfg = load_config(config_path)
        
        assert cfg.model.device == "cpu"
        assert cfg.model.ensemble is False
        # Other fields should be defaults
        assert cfg.model.name == "mst_plus_plus"
        assert cfg.fitting.multiple == 32
    
    def test_load_empty_config(self, tmp_path):
        """Test loading empty config file uses all defaults."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")
        
        cfg = load_config(config_path)
        
        assert isinstance(cfg, RunConfig)
        # All defaults
        assert cfg.model.ensemble is True
        assert cfg.upscaling.enabled is False
    
    def test_regenerate_missing_config(self, tmp_path):
        """Test that missing config is auto-regenerated."""
        config_path = tmp_path / "nonexistent" / "config.yaml"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = load_config(config_path)
            
            # Should have warning
            assert len(w) == 1
            assert "not found" in str(w[0].message).lower()
        
        # Config should exist now
        assert config_path.exists()
        
        # Should be usable
        assert isinstance(cfg, RunConfig)
    
    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises clear error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: {invalid yaml here")
        
        with pytest.raises(ValueError) as exc_info:
            load_config(config_path)
        
        assert "Invalid YAML" in str(exc_info.value)
        assert "Suggestion" in str(exc_info.value)
    
    def test_unknown_keys_warning(self, tmp_path):
        """Test that unknown config keys produce warning."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
model:
  name: mst_plus_plus
unknown_section:
  foo: bar
another_unknown: 123
""")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(config_path)  # Just to trigger the warning
            
            # Should have warning about unknown keys
            unknown_warnings = [x for x in w if "unknown" in str(x.message).lower()]
            assert len(unknown_warnings) == 1
            assert "unknown_section" in str(unknown_warnings[0].message)


class TestCLIOverrides:
    """Tests for CLI override merging."""
    
    def test_no_ensemble_override(self):
        """Test --no-ensemble flag disables ensemble."""
        cfg = RunConfig()
        assert cfg.model.ensemble is True
        
        cfg = merge_cli_overrides(cfg, no_ensemble=True)
        
        assert cfg.model.ensemble is False
    
    def test_upscale_factor_override(self):
        """Test --upscale-factor enables and sets upscaling."""
        cfg = RunConfig()
        assert cfg.upscaling.enabled is False
        
        cfg = merge_cli_overrides(cfg, upscale_factor=4)
        
        assert cfg.upscaling.enabled is True
        assert cfg.upscaling.factor == 4
    
    def test_on_error_override(self):
        """Test --on-error policy override."""
        cfg = RunConfig()
        assert cfg.dataset.on_error == "continue"
        
        cfg = merge_cli_overrides(cfg, on_error="abort")
        
        assert cfg.dataset.on_error == "abort"
    
    def test_multiple_overrides(self):
        """Test multiple overrides at once."""
        cfg = RunConfig()
        
        cfg = merge_cli_overrides(
            cfg,
            no_ensemble=True,
            upscale_factor=2,
            on_error="abort",
        )
        
        assert cfg.model.ensemble is False
        assert cfg.upscaling.enabled is True
        assert cfg.upscaling.factor == 2
        assert cfg.dataset.on_error == "abort"
    
    def test_none_overrides_ignored(self):
        """Test that None values don't override config."""
        cfg = RunConfig()
        original_factor = cfg.upscaling.factor
        
        cfg = merge_cli_overrides(cfg, upscale_factor=None)
        
        # Should not change
        assert cfg.upscaling.factor == original_factor
        assert cfg.upscaling.enabled is False


class TestConfigToDict:
    """Tests for config serialization."""
    
    def test_to_dict_roundtrip(self):
        """Test config can be serialized and deserialized."""
        original = RunConfig()
        
        as_dict = original.to_dict()
        restored = RunConfig.from_dict(as_dict)
        
        assert restored.model.name == original.model.name
        assert restored.model.ensemble == original.model.ensemble
        assert restored.fitting.multiple == original.fitting.multiple
        assert restored.upscaling.factor == original.upscaling.factor
