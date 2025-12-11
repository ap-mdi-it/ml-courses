"""Command-line interface for PDF generation."""

import argparse

from ml_courses.pdf.generator import PDFGenerator


def main():
    """Run the PDF generator CLI."""
    parser = argparse.ArgumentParser(description="Generate PDFs from myst.yml TOC structure")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process the first page",
    )
    parser.add_argument(
        "--base-url",
        default="https://ap-mdi-it.github.io/ml-courses",
        help="Base URL for GitHub Pages site",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PDFs (default: exports/ in repo root)",
    )
    parser.add_argument(
        "--myst-yml",
        default=None,
        help="Path to myst.yml file (default: myst.yml in repo root)",
    )

    args = parser.parse_args()

    generator = PDFGenerator(
        myst_yml_path=args.myst_yml,
        base_url=args.base_url,
        output_dir=args.output_dir,
    )

    print("PDF Generator for myst.yml TOC")
    print("=" * 80)

    if args.test:
        print("\n⚠ TEST MODE: Processing only the first page\n")

    try:
        if args.test:
            # Process only the first file for testing
            generator.load_toc()

            # Clean up existing output directory
            if generator.output_dir.exists():
                print(f"Cleaning up existing directory: {generator.output_dir}")
                import shutil

                shutil.rmtree(generator.output_dir)

            # Create fresh output directory
            generator.output_dir.mkdir(parents=True, exist_ok=True)

            generator.filename_counts = {}  # Reset for test
            first_item = generator.toc[0]
            generator.process_toc_item(first_item, None, 1)
            print(f"\n{'=' * 80}")
            print(f"Test complete! Check the output in: {generator.output_dir}")
            print("To process all pages, run without --test flag.")
            print(f"{'=' * 80}")
        else:
            generator.generate_all_pdfs()
            generator.generate_index()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
