"""PDF generator for myst.yml table of contents.

This module parses the table of contents from myst.yml, generates PDFs for each
page using Playwright/Chromium, and organizes them in a directory structure that
mirrors the TOC hierarchy.

Features:
- Sequential numbering (01_, 02_) to preserve TOC order
- Duplicate filename handling with -1, -2 suffixes (matching myst behavior)
- Flat URL structure with underscores→hyphens conversion
- Title-based directory hierarchy for output organization
"""

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from ml_courses.pdf.converter import html_to_pdf


class PDFGenerator:
    """Generate PDFs from myst.yml TOC structure."""

    def __init__(
        self,
        myst_yml_path: str | Path | None = None,
        base_url: str = "https://ap-mdi-it.github.io/ml-courses",
        output_dir: str | Path | None = None,
    ):
        """Initialize the PDF generator.

        Parameters
        ----------
        myst_yml_path : str | Path | None
            Path to the myst.yml configuration file. If None, looks for myst.yml
            in the repository root (3 levels up from this module).
        base_url : str
            Base URL of the GitHub Pages site
        output_dir : str | Path | None
            Root directory for PDF exports. If None, uses '_exports' in repository root.

        """
        # Set default paths relative to repository root
        # Module is at src/ml_courses/pdf, repo root is 3 levels up
        repo_root = Path(__file__).parent.parent.parent.parent
        self.myst_yml_path = Path(myst_yml_path) if myst_yml_path else (repo_root / "myst.yml")
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else (repo_root / "_exports")
        self.toc: list[dict[str, Any]] | None = None
        # Track seen filenames for duplicate handling (myst adds -1, -2, etc.)
        self.filename_counts: dict[str, int] = {}

    def load_toc(self) -> list[dict[str, Any]]:
        """Load the table of contents from myst.yml.

        Returns
        -------
        list[dict[str, Any]]
            The TOC structure as a list of dictionaries

        Raises
        ------
        ValueError
            If myst.yml is invalid or missing required structure

        """
        with open(self.myst_yml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "project" not in config or "toc" not in config["project"]:
            raise ValueError("Invalid myst.yml: missing project.toc")

        toc = config["project"]["toc"]
        if not isinstance(toc, list):
            raise ValueError("Invalid myst.yml: project.toc must be a list")

        self.toc = toc
        return self.toc

    def get_root_file(self) -> str | None:
        """Get the root file path from TOC.

        The first file entry in the TOC is the root page shown at the base URL.

        Returns
        -------
        str | None
            The file path of the root page, or None if not found
        """
        if self.toc and len(self.toc) > 0:
            first_item = self.toc[0]
            return first_item.get("file") if isinstance(first_item, dict) else None
        return None

    def file_path_to_url(self, file_path: str) -> str:
        """Convert a file path to a GitHub Pages URL.

        Myst builds pages flat under the base URL, using only the filename
        (not the directory structure) and replaces underscores with hyphens.
        For duplicate filenames, myst appends -1, -2, etc.

        Special case: The first file in the TOC is the root page and uses
        the base URL directly.

        Parameters
        ----------
        file_path : str
            File path from myst.yml (e.g., 'book/ml_principles/labs/exploratory_uber_a.ipynb')

        Returns
        -------
        str
            Full URL to the page (e.g., 'https://.../exploratory-uber-a')

        """
        # Special case: first file in TOC is the root page
        root_file = self.get_root_file()
        if root_file and file_path == root_file:
            return self.base_url

        # Extract just the filename without extension
        filename = Path(file_path).stem

        # Replace underscores with hyphens (myst convention)
        filename = filename.replace("_", "-")

        # Handle duplicates (myst adds -1, -2, etc. for subsequent occurrences)
        if filename in self.filename_counts:
            self.filename_counts[filename] += 1
            filename = f"{filename}-{self.filename_counts[filename]}"
        else:
            self.filename_counts[filename] = 0

        # Build full URL (flat structure)
        return f"{self.base_url}/{filename}"

    def file_path_to_output_path(self, file_path: str, title_path: Path, item_number: int) -> Path:
        """Convert a file path to an output PDF path using TOC title hierarchy.

        Parameters
        ----------
        file_path : str
            File path from myst.yml (e.g., 'book/overview.md')
        title_path : Path
            Directory path based on TOC title hierarchy
        item_number : int
            Sequential number for this item to preserve TOC order

        Returns
        -------
        Path
            Path object for the output PDF file

        """
        # Extract just the filename and add numbering prefix
        filename = Path(file_path).stem
        numbered_filename = f"{item_number:02d}_{filename}.pdf"

        # Use the title-based path for directory structure
        return title_path / numbered_filename

    def sanitize_title(self, title: str, item_number: int | None = None) -> str:
        """Sanitize a title for use as a directory name.

        Parameters
        ----------
        title : str
            The title string (may contain emojis and special chars)
        item_number : int | None
            Optional number to prefix for ordering

        Returns
        -------
        str
            Sanitized directory name

        """
        # Remove emojis and special characters
        sanitized = re.sub(r"[^\w\s-]", "", title)
        # Replace spaces with underscores
        sanitized = re.sub(r"\s+", "_", sanitized.strip())
        # Keep original case for readability

        # Add numbering prefix if provided
        if item_number is not None:
            sanitized = f"{item_number:02d}_{sanitized}"

        return sanitized

    def process_toc_item(
        self,
        item: dict[str, Any],
        current_path: Path | None = None,
        item_number: int = 1,
    ) -> int:
        """Recursively process a TOC item and generate PDFs.

        Parameters
        ----------
        item : dict[str, Any]
            A single item from the TOC structure
        current_path : Path | None
            Current directory path based on TOC title hierarchy
        item_number : int
            Sequential number for ordering items

        Returns
        -------
        int
            Next item number to use

        """
        if current_path is None:
            current_path = self.output_dir

        # If item has a file, generate PDF
        if "file" in item:
            file_path = item["file"]
            url = self.file_path_to_url(file_path)
            output_path = self.file_path_to_output_path(file_path, current_path, item_number)

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 80}")
            print(f"Processing: {file_path}")
            print(f"URL: {url}")
            print(f"Output: {output_path}")
            print(f"{'=' * 80}")

            try:
                html_to_pdf(url, str(output_path))
                print(f"✓ Success: {output_path}")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")

            return item_number + 1

        # If item has children, process them recursively
        if "children" in item:
            # If there's a title but no file, create a subdirectory
            if "title" in item and "file" not in item:
                sanitized_title = self.sanitize_title(item["title"], item_number)
                current_path = current_path / sanitized_title
                item_number += 1

            child_number = 1
            for child in item["children"]:
                child_number = self.process_toc_item(child, current_path, child_number)

        return item_number

    def generate_all_pdfs(self) -> None:
        """Generate PDFs for all pages in the TOC."""
        print(f"Loading TOC from {self.myst_yml_path}...")
        self.load_toc()

        # Ensure toc was loaded successfully
        assert self.toc is not None, "TOC must be loaded before generating PDFs"

        print(f"Output directory: {self.output_dir}")
        print(f"Base URL: {self.base_url}")

        # Clean up existing output directory
        if self.output_dir.exists():
            print(f"Cleaning up existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        # Create fresh output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\nStarting PDF generation for all TOC items...")

        # Reset filename counts for duplicate detection
        self.filename_counts = {}

        item_number = 1
        for item in self.toc:
            item_number = self.process_toc_item(item, None, item_number)

        print(f"\n{'=' * 80}")
        print("PDF generation complete!")
        print(f"PDFs saved to: {self.output_dir.absolute()}")
        print(f"{'=' * 80}")

    def generate_index(self) -> None:
        """Generate an index file listing all generated PDFs."""
        pdf_files = sorted(self.output_dir.rglob("*.pdf"))

        index_path = self.output_dir / "index.txt"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("PDF Export Index\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total PDFs: {len(pdf_files)}\n\n")

            for pdf_file in pdf_files:
                rel_path = pdf_file.relative_to(self.output_dir)
                f.write(f"{rel_path}\n")

        print(f"\nIndex generated: {index_path}")
