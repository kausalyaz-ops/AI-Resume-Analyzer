# Resume input handling

This exercise reads PDF and UTF-8 plain-text resumes, normalizes whitespace,
removes standalone page numbers and repeated PDF page-edge headers/footers, and
reports character and whitespace-separated token counts.

The sample files use fictional names and reserved `.example` email addresses;
they contain no real candidate data. Four resumes are PDFs and four are text
files. `scanned.pdf` is an image-only fixture used to demonstrate the no-text
error.

## Run

Install the dependency and run the included samples:

```powershell
python -m pip install -r requirements.txt
python read_resume.py
```

Or pass individual files:

```powershell
python read_resume.py sample_resumes\resume_01.pdf sample_resumes\resume_05.txt
```

`read_resume(path)` returns cleaned text. It raises `ResumeReadError` with a
clear message for unsupported, empty, scanned, or password-protected files.
Missing or inaccessible paths raise the standard `OSError` subclasses.

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```
