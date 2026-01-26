"""Unit tests for VOC XML parser."""

import pytest
from pathlib import Path

from hsi_pipeline.adapters.voc_parser import (
    parse_voc_xml, VOCParseError, VOCAnnotation
)


class TestParseVOCXML:
    """Tests for parse_voc_xml function."""

    def test_parse_valid_xml(self, tmp_path):
        """Should parse valid VOC XML."""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <filename>image.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
            </size>
            <object>
                <name>dog</name>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>50</ymin>
                    <xmax>300</xmax>
                    <ymax>200</ymax>
                </bndbox>
            </object>
        </annotation>
        """
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        
        result = parse_voc_xml(xml_path)
        
        assert isinstance(result, VOCAnnotation)
        assert result.filename == "image.jpg"
        assert result.width == 640
        assert result.height == 480
        assert len(result.bboxes) == 1
        assert result.bboxes[0].xmin == 100
        assert result.bboxes[0].label == "dog"

    def test_multiple_bboxes(self, tmp_path):
        """Should parse multiple objects."""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <filename>image.jpg</filename>
            <size><width>640</width><height>480</height></size>
            <object>
                <name>dog</name>
                <bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox>
            </object>
            <object>
                <name>cat</name>
                <bndbox><xmin>300</xmin><ymin>200</ymin><xmax>400</xmax><ymax>300</ymax></bndbox>
            </object>
        </annotation>
        """
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        
        result = parse_voc_xml(xml_path)
        
        assert len(result.bboxes) == 2
        assert result.bboxes[0].label == "dog"
        assert result.bboxes[1].label == "cat"

    def test_file_not_found_raises(self):
        """Should raise VOCParseError for missing file."""
        with pytest.raises(VOCParseError, match="not found"):
            parse_voc_xml(Path("/nonexistent/file.xml"))

    def test_invalid_xml_raises(self, tmp_path):
        """Should raise VOCParseError for invalid XML."""
        xml_path = tmp_path / "invalid.xml"
        xml_path.write_text("not valid xml <><>")
        
        with pytest.raises(VOCParseError, match="Invalid XML"):
            parse_voc_xml(xml_path)

    def test_missing_filename_raises(self, tmp_path):
        """Should raise VOCParseError for missing filename tag."""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <size><width>640</width><height>480</height></size>
            <object>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
            </object>
        </annotation>
        """
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        
        with pytest.raises(VOCParseError, match="filename"):
            parse_voc_xml(xml_path)

    def test_missing_size_raises(self, tmp_path):
        """Should raise VOCParseError for missing size tag."""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <filename>image.jpg</filename>
            <object>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
            </object>
        </annotation>
        """
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        
        with pytest.raises(VOCParseError, match="size"):
            parse_voc_xml(xml_path)

    def test_no_bboxes_raises(self, tmp_path):
        """Should raise VOCParseError if no valid bboxes found."""
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <filename>image.jpg</filename>
            <size><width>640</width><height>480</height></size>
        </annotation>
        """
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        
        with pytest.raises(VOCParseError, match="No valid bboxes"):
            parse_voc_xml(xml_path)
