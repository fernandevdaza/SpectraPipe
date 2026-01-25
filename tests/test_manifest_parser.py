"""Unit tests for manifest parser."""

import pytest
import json
import yaml

from hsi_pipeline.manifest.parser import (
    parse_manifest,
    ManifestNotFoundError,
    ManifestParseError,
    ManifestValidationError,
)


class TestParseManifest:
    """Tests for parse_manifest function."""

    def test_parse_yaml_manifest(self, tmp_path):
        """Should parse valid YAML manifest."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img1.png").touch()
        (root_dir / "img2.png").touch()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [
                {"id": "s1", "image": "img1.png"},
                {"id": "s2", "image": "img2.png"},
            ]
        }
        
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        result = parse_manifest(manifest_path)
        
        assert len(result.samples) == 2
        assert result.samples[0].id == "s1"
        assert result.samples[1].id == "s2"

    def test_parse_json_manifest(self, tmp_path):
        """Should parse valid JSON manifest."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img1.png").touch()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [{"id": "s1", "image": "img1.png"}]
        }
        
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
        
        result = parse_manifest(manifest_path)
        
        assert len(result.samples) == 1
        assert result.samples[0].id == "s1"

    def test_manifest_not_found_raises(self):
        """Should raise ManifestNotFoundError for missing file."""
        with pytest.raises(ManifestNotFoundError, match="not found"):
            parse_manifest("/nonexistent/manifest.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        """Should raise ManifestParseError for invalid YAML."""
        manifest_path = tmp_path / "bad.yaml"
        manifest_path.write_text("root: [unbalanced")
        
        with pytest.raises(ManifestParseError, match="YAML"):
            parse_manifest(manifest_path)

    def test_invalid_json_raises(self, tmp_path):
        """Should raise ManifestParseError for invalid JSON."""
        manifest_path = tmp_path / "bad.json"
        manifest_path.write_text("{invalid json}")
        
        with pytest.raises(ManifestParseError, match="JSON"):
            parse_manifest(manifest_path)

    def test_unsupported_format_raises(self, tmp_path):
        """Should raise ManifestParseError for unsupported format."""
        manifest_path = tmp_path / "bad.txt"
        manifest_path.write_text("root: /tmp")
        
        with pytest.raises(ManifestParseError, match="Unsupported"):
            parse_manifest(manifest_path)

    def test_missing_root_raises(self, tmp_path):
        """Should raise ManifestValidationError for missing root."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("samples: []")
        
        with pytest.raises(ManifestValidationError, match="root"):
            parse_manifest(manifest_path)

    def test_nonexistent_root_raises(self, tmp_path):
        """Should raise ManifestValidationError for nonexistent root."""
        manifest_data = {"root": "/nonexistent/path", "samples": []}
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="not found"):
            parse_manifest(manifest_path)

    def test_missing_samples_and_pattern_raises(self, tmp_path):
        """Should raise ManifestValidationError if neither samples nor pattern."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        
        manifest_data = {"root": str(root_dir)}
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="samples.*pattern"):
            parse_manifest(manifest_path)

    def test_both_samples_and_pattern_raises(self, tmp_path):
        """Should raise ManifestValidationError if both samples and pattern."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [],
            "pattern": "*.png"
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="both"):
            parse_manifest(manifest_path)

    def test_sample_missing_id_raises(self, tmp_path):
        """Should raise ManifestValidationError for sample without id."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [{"image": "img.png"}]
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="id"):
            parse_manifest(manifest_path)

    def test_sample_missing_image_raises(self, tmp_path):
        """Should raise ManifestValidationError for sample without image."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [{"id": "s1"}]
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="image"):
            parse_manifest(manifest_path)

    def test_duplicate_ids_raises(self, tmp_path):
        """Should raise ManifestValidationError for duplicate sample IDs."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img1.png").touch()
        (root_dir / "img2.png").touch()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [
                {"id": "s1", "image": "img1.png"},
                {"id": "s1", "image": "img2.png"},
            ]
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="Duplicate"):
            parse_manifest(manifest_path)

    def test_pattern_discovery(self, tmp_path):
        """Should discover samples using glob pattern."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img1.png").touch()
        (root_dir / "img2.png").touch()
        (root_dir / "other.txt").touch()
        
        manifest_data = {
            "root": str(root_dir),
            "pattern": "*.png"
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        result = parse_manifest(manifest_path)
        
        assert len(result.samples) == 2
        ids = {s.id for s in result.samples}
        assert "img1" in ids
        assert "img2" in ids

    def test_pattern_no_matches_raises(self, tmp_path):
        """Should raise ManifestValidationError if pattern matches nothing."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        
        manifest_data = {
            "root": str(root_dir),
            "pattern": "*.png"
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        with pytest.raises(ManifestValidationError, match="matched no files"):
            parse_manifest(manifest_path)

    def test_sample_with_roi_mask(self, tmp_path):
        """Should parse sample with roi_mask."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img.png").touch()
        (root_dir / "mask.png").touch()
        
        manifest_data = {
            "root": str(root_dir),
            "samples": [
                {"id": "s1", "image": "img.png", "roi_mask": "mask.png"}
            ]
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        result = parse_manifest(manifest_path)
        
        assert result.samples[0].roi_mask == "mask.png"
        assert result.samples[0].roi_mask_resolved is not None

    def test_relative_root_resolved(self, tmp_path):
        """Should resolve relative root path relative to manifest."""
        root_dir = tmp_path / "dataset"
        root_dir.mkdir()
        (root_dir / "img.png").touch()
        
        manifest_data = {
            "root": "dataset",
            "samples": [{"id": "s1", "image": "img.png"}]
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)
        
        result = parse_manifest(manifest_path)
        
        assert result.root == root_dir.resolve()
