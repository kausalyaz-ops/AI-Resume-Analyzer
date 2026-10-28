"""Read and report text from PDF and plain-text resume files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import pymupdf as fitz


class ResumeReadError(ValueError):
    """Raised when a resume cannot be read as usable text."""


_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line and not _PAGE_NUMBER.fullmatch(line):
            lines.append(line)
    return lines


def _clean_pdf_pages(page_texts: list[str]) -> str:
    page_lines = [_clean_lines(page_text) for page_text in page_texts]
    repeated_edge_lines: set[str] = set()

    if len(page_lines) > 1:
        edge_pages: dict[str, set[int]] = {}
        for page_index, lines in enumerate(page_lines):
            for line in set(lines[:3] + lines[-2:]):
                edge_pages.setdefault(line.casefold(), set()).add(page_index)
        repeated_edge_lines = {
            line for line, pages in edge_pages.items() if len(pages) > 1
        }

    cleaned_pages: list[str] = []
    for lines in page_lines:
        filtered = [line for line in lines if line.casefold() not in repeated_edge_lines]
        if filtered:
            cleaned_pages.append("\n".join(filtered))
    return "\n".join(cleaned_pages).strip()


def clean_resume_text(text: str) -> str:
    """Collapse extra whitespace and remove standalone page numbers."""
    return "\n".join(_clean_lines(text)).strip()


def read_resume(path: str | Path) -> str:
    """Read a PDF or UTF-8 text resume and return cleaned text.

    Raises:
        ResumeReadError: If the file type is unsupported or contains no text.
        OSError: If the path cannot be read.
    """
    resume_path = Path(path)
    suffix = resume_path.suffix.lower()

    if suffix == ".txt":
        text = clean_resume_text(resume_path.read_text(encoding="utf-8-sig"))
        if not text:
            raise ResumeReadError("Error - no text found (empty file)")
        return text

    if suffix != ".pdf":
        raise ResumeReadError(
            f"Unsupported file type: {suffix or '(no extension)'}. Use .pdf or .txt."
        )

    try:
        with fitz.open(resume_path) as document:
            if document.needs_pass:
                raise ResumeReadError("Error - PDF is password-protected")
            page_texts = [page.get_text("text") for page in document]
            has_images = any(page.get_images(full=True) for page in document)
    except fitz.FileDataError as error:
        raise ResumeReadError("Error - unable to read PDF file") from error

    text = _clean_pdf_pages(page_texts)
    if not text:
        message = (
            "Error - no text found (scanned file)"
            if has_images
            else "Error - no text found (empty PDF)"
        )
        raise ResumeReadError(message)
    return text


def count_tokens(text: str) -> int:
    """Count whitespace-separated tokens for a simple, reproducible estimate."""
    return len(text.split())


def _sample_paths() -> list[Path]:
    sample_dir = Path(__file__).parent / "sample_resumes"
    paths = sorted(sample_dir.glob("resume_*"))
    scanned_sample = sample_dir / "scanned.pdf"
    if scanned_sample.exists():
        paths.append(scanned_sample)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and count text in PDF and TXT resumes."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Resume files to inspect (defaults to the included fictional samples).",
    )
    args = parser.parse_args(argv)
    paths = args.paths or _sample_paths()

    for path in paths:
        try:
            text = read_resume(path)
        except (OSError, ResumeReadError) as error:
            print(f"{path.name}: {error}")
        else:
            print(
                f"{path.name}: {len(text)} characters, "
                f"{count_tokens(text)} tokens"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
