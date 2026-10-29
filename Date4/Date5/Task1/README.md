# Resume Analyzer API

FastAPI service that accepts one resume as a PDF, UTF-8 TXT upload, or plain
text, plus a job description. It returns extracted candidate information,
job-match score, token usage, and an estimated API cost.

## Setup

From the repository root in PowerShell:

```powershell
python -m pip install -r Date4\Date5\Task1\requirements.txt
Copy-Item Date4\Date5\Task1\.env.example .env
```

Edit `.env` locally and set `OPENAI_API_KEY`. The key is loaded from the
repository-root `.env` (or an environment variable), is never part of API
responses, and is never logged. `.env` is ignored by Git.

The default model is `gpt-4o-mini`. The default estimated rates are
`$0.15/1M` input tokens and `$0.60/1M` output tokens; configure
`OPENAI_INPUT_USD_PER_1M_TOKENS` and
`OPENAI_OUTPUT_USD_PER_1M_TOKENS` in `.env` for current provider pricing. The
returned cost is an estimate, not a billing record.

## Run and interactive API docs

```powershell
python -m uvicorn Date4.Date5.Task1.app:app --reload
```

- Health check: `GET http://127.0.0.1:8000/health`
- Swagger UI and interactive request example: `http://127.0.0.1:8000/docs`

`POST /analyze` uses `multipart/form-data`. Provide `job_description` and
exactly one of:

- `resume_file`: PDF (`application/pdf`) or UTF-8 TXT (`text/plain`), maximum
  5 MiB; or
- `resume_text`: plain text, maximum 5 MiB encoded as UTF-8.

Example request with a text resume:

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze `
  -F "resume_text=Alex Example; Python, SQL, REST APIs, and testing experience" `
  -F "job_description=Python Developer: build Python services, REST APIs, and SQL-backed features"
```

Invalid input returns a clear `4xx`: unsupported file type is `415`, oversized
resume is `413`, and malformed/empty input is `422`. Missing server credentials
returns `503`; upstream provider failures return `502`.

The response includes candidate extraction, score, matched/missing skills,
reasoning, provider token usage, and estimated USD cost. Treat the score as
decision support only; a human should review evidence and context.

Run tests without making model requests:

```powershell
python -m unittest discover -s Date4\Date5\Task1\tests -v
```
