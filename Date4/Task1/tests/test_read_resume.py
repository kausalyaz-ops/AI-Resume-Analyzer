import tempfile
import unittest
from pathlib import Path
import sys

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from read_resume import ResumeReadError, clean_resume_text, count_tokens, read_resume


class ReadResumeTests(unittest.TestCase):
    def test_reads_and_cleans_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.txt"
            path.write_text("  Alex   Example  \n\nPage 2\nAnalyst   role \n", encoding="utf-8")

            self.assertEqual(read_resume(path), "Alex Example\nAnalyst role")

    def test_reads_pdf_and_removes_repeated_headers_and_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pdf"
            document = fitz.open()
            for body, page_number in (("First page", "1"), ("Second page", "2")):
                page = document.new_page()
                page.insert_text((72, 72), "ALEX EXAMPLE")
                page.insert_text((72, 86), "Software Engineer")
                page.insert_text((72, 100), "alex@example.test")
                page.insert_text((72, 128), body)
                page.insert_text((72, 740), page_number)
            document.save(path)
            document.close()

            self.assertEqual(read_resume(path), "First page\nSecond page")

    def test_scanned_pdf_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scanned.pdf"
            source = fitz.open()
            source_page = source.new_page()
            source_page.insert_text((72, 72), "FICTIONAL SCANNED RESUME")
            image = source_page.get_pixmap().tobytes("png")
            scanned = fitz.open()
            scanned_page = scanned.new_page()
            scanned_page.insert_image(scanned_page.rect, stream=image)
            scanned.save(path)
            source.close()
            scanned.close()

            with self.assertRaisesRegex(
                ResumeReadError, "no text found \\(scanned file\\)"
            ):
                read_resume(path)

    def test_empty_text_file_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("  \n Page 1 \n", encoding="utf-8")

            with self.assertRaisesRegex(ResumeReadError, "empty file"):
                read_resume(path)

    def test_rejects_unsupported_file_type(self) -> None:
        with self.assertRaisesRegex(ResumeReadError, "Use .pdf or .txt"):
            read_resume("resume.docx")

    def test_reports_whitespace_separated_token_count(self) -> None:
        self.assertEqual(count_tokens("one  two\nthree"), 3)

    def test_cleans_whitespace_and_page_number_lines(self) -> None:
        self.assertEqual(clean_resume_text("  A   B\nPage 1 of 2\n\nC "), "A B\nC")


if __name__ == "__main__":
    unittest.main()
