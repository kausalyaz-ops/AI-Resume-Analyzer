"""Extract structured resume data using OpenAI JSON Schema response mode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TASK2_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK2_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Date4.Task1.read_resume import read_resume  # noqa: E402


SCHEMA_PATH = TASK2_DIR / "schema_v1.json"
PROMPT_PATH = TASK2_DIR / "system_prompt_v1.md"
RESUME_DIR = REPO_ROOT / "Date4" / "Task1" / "sample_resumes"
API_URL = "https://api.openai.com/v1/chat/completions"
SCHEMA_NAME = "resume_extraction_v1"
REQUIRED_FIELDS = {
    "candidate_name",
    "skills",
    "years_of_experience",
    "education",
    "strengths",
    "gaps",
}


def validate_extraction(result: Any, resume_text: str) -> None:
    """Raise ValueError when a model result violates the output contract."""
    if not isinstance(result, dict) or set(result) != REQUIRED_FIELDS:
        raise ValueError("Extraction must contain exactly the required schema fields")

    candidate_name = result["candidate_name"]
    if candidate_name is not None and not isinstance(candidate_name, str):
        raise ValueError("candidate_name must be a string or null")

    years = result["years_of_experience"]
    if years is not None and (
        isinstance(years, bool) or not isinstance(years, (int, float)) or years < 0
    ):
        raise ValueError("years_of_experience must be a non-negative number or null")

    for field in ("education", "strengths", "gaps"):
        values = result[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{field} must be an array of non-empty strings")

    skills = result["skills"]
    if not isinstance(skills, list):
        raise ValueError("skills must be an array")
    for skill in skills:
        if (
            not isinstance(skill, dict)
            or set(skill) != {"name", "evidence"}
            or not isinstance(skill["name"], str)
            or not skill["name"].strip()
            or not isinstance(skill["evidence"], str)
            or not skill["evidence"].strip()
            or skill["evidence"] not in resume_text
        ):
            raise ValueError(
                "Each skill must have a name and verbatim evidence from the resume"
            )


def _extract_one(
    resume_name: str,
    resume_text: str,
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": schema,
            },
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Extract the resume named {resume_name}:\n\n{resume_text}",
            },
        ],
    }
    request = Request(
        API_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"Extraction request for {resume_name} failed with HTTP {error.code}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect while extracting {resume_name}: {error.reason}"
        ) from error

    try:
        content = response_body["choices"][0]["message"]["content"]
        extraction = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Model returned an invalid JSON response for {resume_name}"
        ) from error

    try:
        validate_extraction(extraction, resume_text)
    except ValueError as error:
        raise RuntimeError(f"Invalid extraction for {resume_name}: {error}") from error
    return extraction


def run_extractions(
    resume_paths: list[Path], *, api_key: str, model: str
) -> dict[str, Any]:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for path in resume_paths:
        resume_text = read_resume(path)
        extraction = _extract_one(
            path.name,
            resume_text,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            schema=schema,
        )
        results.append({"resume": path.name, "extraction": extraction})
    return {"prompt_version": "v1", "model": model, "temperature": 0, "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract all eight fictional sample resumes using JSON Schema mode."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK2_DIR / "results_v1.json",
        help="Output JSON file (default: results_v1.json).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model (default: OPENAI_MODEL or gpt-4o-mini).",
    )
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error(
            "OPENAI_API_KEY is not set; configure it in the environment without "
            "putting the key in source control."
        )

    resume_paths = sorted(RESUME_DIR.glob("resume_*"))
    if len(resume_paths) != 8:
        parser.error(
            f"Expected exactly 8 sample resumes in {RESUME_DIR}, found {len(resume_paths)}."
        )

    results = run_extractions(resume_paths, api_key=api_key, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(results['results'])} resumes with {args.model} at temperature 0.")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
