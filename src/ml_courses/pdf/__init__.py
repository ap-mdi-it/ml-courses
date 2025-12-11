"""PDF generation module for ml-courses.

This module provides tools for generating PDF versions of course materials
from the published GitHub Pages site using Playwright/Chromium.
"""

from ml_courses.pdf.converter import html_to_pdf
from ml_courses.pdf.generator import PDFGenerator

__all__ = ["PDFGenerator", "html_to_pdf"]
