"""Module for converting HTML files and URLs to PDF format.

This module provides functionality to convert HTML files or web URLs to PDF
documents using Playwright's Chromium browser automation.
"""

import os

from playwright.sync_api import sync_playwright


def html_to_pdf(input_path: str, output_path: str) -> None:
    """Convert an HTML file or URL to PDF using Playwright.

    Parameters
    ----------
    input_path : str
        URL or file path to convert
    output_path : str
        Path where the PDF will be saved

    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Check if input is a local file or URL
        if input_path.startswith("http"):
            url = input_path
        else:
            # Convert local path to file:// URL
            abs_path = os.path.abspath(input_path)
            url = f"file://{abs_path}"

        print(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")

        # Add some options for better PDF output (A4, print background, etc.)
        print(f"Generating PDF at {output_path}...")
        page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"},
        )

        browser.close()
        print("Done.")
