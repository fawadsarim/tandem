"""Convert a Jupyter notebook (including its saved cell outputs) to a PDF.

Renders via a headless Chromium browser (nbconvert's WebPDF exporter), which
avoids needing a full LaTeX toolchain -- dev machines rarely have one, and
installing one just for this is a multi-GB detour. One-time setup:

    pip install "nbconvert[webpdf]"
    playwright install chromium

Usage:
    python scripts/notebook_to_pdf.py <notebook.ipynb> [output.pdf]
"""
import asyncio
import sys
from pathlib import Path

import nbformat
from nbconvert import WebPDFExporter


def convert(ipynb_path: Path, pdf_path: Path) -> None:
    nb = nbformat.read(ipynb_path, as_version=4)
    exporter = WebPDFExporter()
    body, _ = exporter.from_notebook_node(nb)
    pdf_path.write_bytes(body)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python notebook_to_pdf.py <notebook.ipynb> [output.pdf]")
    ipynb_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ipynb_path.with_suffix(".pdf")
    if sys.platform == "win32":
        # Playwright's subprocess transport needs the Proactor loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    convert(ipynb_path, pdf_path)
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
