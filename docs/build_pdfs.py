"""Build the two PDFs in docs/ from their sources.

    docs/OBJECTIVES_PLAN.md            -> docs/OBJECTIVES_PLAN.pdf
    docs/STAIR_Project_Overview.html   -> docs/STAIR_Project_Overview.pdf

Run it on the laptop, from the repo folder:
    .venv\\Scripts\\python.exe docs\\build_pdfs.py

The Markdown is converted to HTML with markdown-it-py. Microsoft Edge, in headless mode,
then prints each page to PDF; page footers come from CSS @page margin boxes. Set the
STAIR_EDGE environment variable if Edge is somewhere unusual.
"""

import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = Path(__file__).resolve().parent
PLAN_SOURCE = DOCS / "OBJECTIVES_PLAN.md"
PLAN_TARGET = DOCS / "OBJECTIVES_PLAN.pdf"
OVERVIEW_SOURCE = DOCS / "STAIR_Project_Overview.html"
OVERVIEW_TARGET = DOCS / "STAIR_Project_Overview.pdf"

PLAN_TITLE = "STAIR for 10-K filings: objectives plan"
PLAN_FOOTER = "STAIR for 10-K filings · Objectives plan"

PLAN_CSS = """
@page {
  size: Letter;
  margin: 0.75in 0.85in 0.85in 0.85in;
  @bottom-left { content: "%(footer)s"; font: 8.5pt Calibri, sans-serif; color: #6B6F86; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8.5pt Calibri, sans-serif; color: #6B6F86; }
}
body { font-family: Calibri, Carlito, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1C1E2E; }
.kicker { font-weight: bold; font-size: 9pt; letter-spacing: 0.04em; text-transform: uppercase; color: #C8811A; margin: 0 0 4pt; }
h1, h2, h3 { font-family: Cambria, Georgia, serif; color: #2B2F5E; break-after: avoid; }
h1 { font-size: 24pt; margin: 0 0 10pt; padding-bottom: 6pt; border-bottom: 1.2pt solid #2B2F5E; }
h2 { font-size: 16pt; margin: 18pt 0 6pt; }
h3 { font-size: 12.5pt; margin: 12pt 0 4pt; }
strong { color: #2B2F5E; }
p { margin: 4pt 0 6pt; }
ul, ol { padding-left: 18pt; margin: 4pt 0 6pt; }
li { margin: 2pt 0; }
hr { border: 0; border-top: 0.75pt solid #D6D5E0; margin: 18pt 0; }
a { color: #2B2F5E; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #F0EFF6; padding: 0 2pt; border-radius: 2pt; }
pre { background: #F0EFF6; padding: 8pt 10pt; border-radius: 4pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%%; font-size: 8.5pt; margin: 6pt 0 10pt; }
th { background: #2B2F5E; color: #FFFFFF; text-align: left; padding: 5pt 6pt; }
td { padding: 5pt 6pt; border-bottom: 0.5pt solid #D6D5E0; vertical-align: top; }
th strong, th code { color: #FFFFFF; background: none; }
td code { font-size: 8pt; }
tr { break-inside: avoid; }
"""


def find_edge() -> str:
    candidates = [
        os.environ.get("STAIR_EDGE"),
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    sys.exit("Microsoft Edge not found. Set STAIR_EDGE to msedge.exe's path.")


def print_to_pdf(page_path: Path, target: Path, tmp: Path) -> None:
    before = target.stat().st_mtime if target.exists() else 0
    subprocess.run(
        [
            find_edge(),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={tmp / 'profile'}",
            f"--print-to-pdf={target}",
            page_path.as_uri(),
        ],
        check=True,
        timeout=180,
        capture_output=True,
    )
    if not target.exists() or target.stat().st_mtime <= before:
        sys.exit(f"Edge did not write {target.name}.")
    print(f"Wrote {target.relative_to(DOCS.parent)}")


def plan_page() -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(PLAN_SOURCE.read_text(encoding="utf-8"))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(PLAN_TITLE)}</title>"
        f"<style>{PLAN_CSS % {'footer': PLAN_FOOTER}}</style></head><body>"
        f"<p class='kicker'>{html.escape(PLAN_FOOTER)}</p>{body}</body></html>"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plan_html = tmp / "plan.html"
        plan_html.write_text(plan_page(), encoding="utf-8")
        print_to_pdf(plan_html, PLAN_TARGET, tmp)
        print_to_pdf(OVERVIEW_SOURCE, OVERVIEW_TARGET, tmp)


if __name__ == "__main__":
    main()
