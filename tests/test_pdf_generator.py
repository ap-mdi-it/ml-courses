"""Tests for PDF generation functionality."""

from pathlib import Path

import pytest

from ml_courses.pdf.generator import PDFGenerator


@pytest.fixture
def generator():
    """Create a PDF generator instance for testing."""
    return PDFGenerator()


@pytest.fixture
def repo_root():
    """Get the repository root path."""
    # Tests are in tests/, repo root is parent
    return Path(__file__).parent.parent


def test_load_toc(generator):
    """Test that TOC can be loaded from myst.yml."""
    toc = generator.load_toc()
    assert isinstance(toc, list)
    assert len(toc) > 0
    # Check that the first item has expected structure
    assert "file" in toc[0] or "title" in toc[0]


def test_get_root_file(generator):
    """Test that root file is correctly identified from TOC."""
    generator.load_toc()
    root_file = generator.get_root_file()
    assert root_file == "book/overview.md"


def test_file_path_to_url(generator):
    """Test conversion of file paths to URLs.

    Myst builds pages flat under the base URL with underscores replaced by hyphens.
    Tests also verify duplicate handling with -1, -2 suffixes.
    The first file in TOC (book/overview.md) is the root page at base URL.
    """
    # Load TOC to identify root file
    generator.load_toc()

    # Reset filename counts for clean test
    generator.filename_counts = {}

    test_cases = [
        # First file in TOC is the root page (no filename appended)
        ("book/overview.md", "https://ap-mdi-it.github.io/ml-courses"),
        (
            "book/ml_principles/basics/training.ipynb",
            "https://ap-mdi-it.github.io/ml-courses/training",
        ),
        (
            "book/ml_principles/labs/exploratory_uber_a.ipynb",
            "https://ap-mdi-it.github.io/ml-courses/exploratory-uber-a",
        ),
        (
            "book/math_foundations/linalg/tensors.ipynb",
            "https://ap-mdi-it.github.io/ml-courses/tensors",
        ),
    ]

    for file_path, expected_url in test_cases:
        url = generator.file_path_to_url(file_path)
        assert url == expected_url

    # Test duplicate handling
    generator.filename_counts = {}
    url1 = generator.file_path_to_url("book/section1/glossary.md")
    url2 = generator.file_path_to_url("book/section2/glossary.md")
    url3 = generator.file_path_to_url("book/section3/glossary.md")
    assert url1 == "https://ap-mdi-it.github.io/ml-courses/glossary"
    assert url2 == "https://ap-mdi-it.github.io/ml-courses/glossary-1"
    assert url3 == "https://ap-mdi-it.github.io/ml-courses/glossary-2"


def test_file_path_to_output_path(generator, repo_root):
    """Test conversion of file paths to output PDF paths using title hierarchy.

    Tests verify that item numbering is properly applied to output filenames.
    """
    from pathlib import Path as P

    test_cases = [
        ("book/overview.md", P("_exports"), 1, "01_overview.pdf"),
        (
            "book/ml_principles/basics/training.ipynb",
            P("_exports/ml_principles/basics"),
            3,
            "03_training.pdf",
        ),
        (
            "book/ml_principles/basics/testing.ipynb",
            P("_exports/ml_principles/basics"),
            15,
            "15_testing.pdf",
        ),
    ]

    for file_path, title_path, item_number, expected_filename in test_cases:
        output_path = generator.file_path_to_output_path(file_path, title_path, item_number)
        # Check that the filename matches with numbering prefix
        assert output_path.name == expected_filename
        # Check that it's under the title path
        assert output_path.parent == title_path


def test_sanitize_title(generator):
    """Test title sanitization for directory names.

    Tests verify both plain sanitization and optional numbering prefix.
    Case is preserved for readability.
    """
    test_cases = [
        ("🛈 Algemene informatie", None, "Algemene_informatie"),
        ("⚡ ML Principles", None, "ML_Principles"),
        ("∑ Mathematical Foundations", None, "Mathematical_Foundations"),
        ("Labo's", None, "Labos"),
        ("🛈 Algemene informatie", 1, "01_Algemene_informatie"),
        ("⚡ ML Principles", 5, "05_ML_Principles"),
        ("∑ Mathematical Foundations", 12, "12_Mathematical_Foundations"),
    ]

    for title, item_number, expected in test_cases:
        sanitized = generator.sanitize_title(title, item_number)
        assert sanitized == expected


def test_generator_with_custom_paths(tmp_path):
    """Test generator with custom output directory."""
    output_dir = tmp_path / "custom_output"
    generator = PDFGenerator(output_dir=output_dir)
    assert generator.output_dir == output_dir


def test_invalid_myst_yml(tmp_path):
    """Test that invalid myst.yml raises appropriate error."""
    # Create an invalid myst.yml
    invalid_yml = tmp_path / "invalid.yml"
    invalid_yml.write_text("invalid: yaml\nwithout: toc\n")

    generator = PDFGenerator(myst_yml_path=invalid_yml)
    with pytest.raises(ValueError, match=r"Invalid myst.yml"):
        generator.load_toc()
