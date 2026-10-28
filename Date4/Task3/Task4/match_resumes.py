"""Match the eight fictional resumes to two jobs and compare repeat scores."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TASK4_DIR = Path(__file__).resolve().parent
TASK1_DIR = TASK4_DIR.parents[1] / "Task1"
RESUME_DIR = TASK1_DIR / "sample_resumes"
JOBS_DIR = TASK4_DIR / "jobs"
PROMPT_PATH = TASK4_DIR / "matching_prompt_v1.md"
API_URL = "https://api.openai.com/v1/chat/completions"
LOGGER = logging.getLogger("resume_job_matching")
Job = dict[str, Any]
Match = dict[str, Any]
Matcher = Callable[[str, str, Job], tuple[Match, dict[str, int | None]]]
MATCH_FIELDS = {"score", "matched_skills", "missing_skills", "reasoning"}
SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "matched_skills", "missing_skills", "reasoning"],
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "matched_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
}


def validate_match(result: Any, job: Job) -> None:
    """Validate the model's output shape and skill lists against the job."""
    if not isinstance(result, dict) or set(result) != MATCH_FIELDS:
        raise ValueError("Output must contain exactly score, matched_skills, missing_skills, reasoning")
    if (
        not isinstance(result["score"], int)
        or isinstance(result["score"], bool)
        or not 0 <= result["score"] <= 100
    ):
        raise ValueError("score must be an integer from 0 through 100")
    for field in ("matched_skills", "missing_skills"):
        values = result[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{field} must be an array of non-empty strings")
    if not isinstance(result["reasoning"], str) or not result["reasoning"].strip():
        raise ValueError("reasoning must be a non-empty string")

    required = {skill.casefold() for skill in job["required_skills"]}
    matched = {skill.casefold() for skill in result["matched_skills"]}
    missing = {skill.casefold() for skill in result["missing_skills"]}
    if matched - required or missing - required:
        raise ValueError("matched_skills and missing_skills must use listed job requirements")
    if matched & missing:
        raise ValueError("A required skill cannot be both matched and missing")


def keyword_match_score(resume_text: str, required_skills: list[str]) -> int:
    """Return a reproducible exact-term coverage score for job requirements."""
    if not required_skills:
        return 0
    searchable = resume_text.casefold()
    matched = 0
    for skill in required_skills:
        term = skill.casefold()
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, searchable):
            matched += 1
    return round(matched * 100 / len(required_skills))


def _request_match(
    resume_name: str,
    resume_text: str,
    job: Job,
    *,
    api_key: str,
    model: str,
    system_prompt: str,
) -> tuple[Match, dict[str, int | None]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Job description:\n{job['description']}\n\n"
                f"Required skills:\n{json.dumps(job['required_skills'])}\n\n"
                f"Resume ({resume_name}):\n{resume_text}"
            ),
        },
    ]
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "resume_job_match",
                "strict": True,
                "schema": SCORE_SCHEMA,
            },
        },
        "messages": messages,
    }
    request = Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
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
            f"Matching request for {resume_name}/{job['job_id']} failed (HTTP {error.code})"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect for {resume_name}/{job['job_id']}: {error.reason}"
        ) from error
    try:
        message = response_body["choices"][0]["message"]
        if message.get("refusal"):
            raise RuntimeError(f"Model refused {resume_name}/{job['job_id']}")
        result = json.loads(message["content"])
        usage_body = response_body.get("usage") or {}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid model response for {resume_name}/{job['job_id']}"
        ) from error
    validate_match(result, job)
    usage: dict[str, int | None] = {
        "prompt_tokens": usage_body.get("prompt_tokens"),
        "completion_tokens": usage_body.get("completion_tokens"),
        "total_tokens": usage_body.get("total_tokens"),
    }
    return result, usage


def run_matchings(
    resumes: list[tuple[str, str]],
    jobs: list[Job],
    matcher: Matcher,
) -> dict[str, Any]:
    """Run each resume/job pair twice and rank by the mean LLM score."""
    by_job: list[dict[str, Any]] = []
    for job in jobs:
        candidates: list[dict[str, Any]] = []
        for resume_name, resume_text in resumes:
            runs: list[dict[str, Any]] = []
            for run_number in (1, 2):
                result, usage = matcher(resume_name, resume_text, job)
                validate_match(result, job)
                runs.append({"run": run_number, **result, "token_usage": usage})
                LOGGER.info(
                    "resume=%s job=%s run=%d temperature=0 score=%d "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    resume_name,
                    job["job_id"],
                    run_number,
                    result["score"],
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                )
            first_score, second_score = runs[0]["score"], runs[1]["score"]
            candidates.append(
                {
                    "candidate_id": Path(resume_name).stem,
                    "resume": resume_name,
                    "runs": runs,
                    "mean_score": round((first_score + second_score) / 2, 1),
                    "score_delta": abs(first_score - second_score),
                    "consistent": first_score == second_score,
                    "keyword_score": keyword_match_score(
                        resume_text, job["required_skills"]
                    ),
                    "matched_skills": runs[0]["matched_skills"],
                    "missing_skills": runs[0]["missing_skills"],
                    "reasoning": runs[0]["reasoning"],
                }
            )
        candidates.sort(key=lambda candidate: (-candidate["mean_score"], candidate["candidate_id"]))
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
        by_job.append(
            {
                "job_id": job["job_id"],
                "job_title": job["title"],
                "required_skills": job["required_skills"],
                "candidates": candidates,
            }
        )
    return {
        "prompt_version": "v1",
        "temperature": 0,
        "repetitions_per_pair": 2,
        "ranking_method": "mean of two LLM scores; tie-break by resume ID",
        "jobs": by_job,
    }


def _load_jobs() -> list[Job]:
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        for field in ("job_id", "title", "description", "required_skills"):
            if field not in job:
                raise ValueError(f"{path.name} is missing {field}")
        jobs.append(job)
    if len(jobs) != 2:
        raise ValueError(f"Expected exactly 2 job descriptions, found {len(jobs)}")
    return jobs


def _load_resumes() -> list[tuple[str, str]]:
    import sys

    sys.path.insert(0, str(TASK1_DIR.parent.parent))
    from Date4.Task1.read_resume import read_resume

    paths = sorted(RESUME_DIR.glob("resume_*"))
    if len(paths) != 8:
        raise ValueError(f"Expected exactly 8 resumes, found {len(paths)}")
    return [(path.name, read_resume(path)) for path in paths]


def _render_report(results: dict[str, Any], model: str) -> str:
    lines = [
        "# Resume-to-job ranking results",
        "",
        f"Prompt: v1 | Model: {model} | Temperature: 0 | Repetitions per pair: 2",
        "",
        "Ranks use the mean of the two LLM scores; ties use resume filename. "
        "Keyword score is an exact required-term coverage baseline, not an LLM score.",
        "",
    ]
    for job in results["jobs"]:
        lines.extend(
            [
                f"## {job['job_title']}",
                "",
                "| Rank | Resume | LLM run 1 | LLM run 2 | Mean | Delta | Consistent | Keyword | Matched skills | Missing skills |",
                "|---:|---|---:|---:|---:|---:|:---:|---:|---|---|",
            ]
        )
        for candidate in job["candidates"]:
            run1, run2 = candidate["runs"]
            matched = ", ".join(candidate["matched_skills"]) or "None"
            missing = ", ".join(candidate["missing_skills"]) or "None"
            lines.append(
                f"| {candidate['rank']} | {candidate['resume']} | {run1['score']} | "
                f"{run2['score']} | {candidate['mean_score']:.1f} | "
                f"{candidate['score_delta']} | "
                f"{'Yes' if candidate['consistent'] else 'No'} | "
                f"{candidate['keyword_score']} | {matched} | {missing} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Fairness note",
            "",
            *(f"- {line}" for line in (TASK4_DIR / "fairness_note.md").read_text(
                encoding="utf-8"
            ).splitlines() if line.strip("- ")),
            "",
        ]
    )
    return "\n".join(lines)


def _run_keyword_baseline(resumes: list[tuple[str, str]], jobs: list[Job]) -> dict[str, Any]:
    baseline = []
    for job in jobs:
        candidates = [
            {
                "candidate_id": Path(name).stem,
                "resume": name,
                "keyword_score": keyword_match_score(text, job["required_skills"]),
            }
            for name, text in resumes
        ]
        candidates.sort(
            key=lambda candidate: (-candidate["keyword_score"], candidate["candidate_id"])
        )
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank
        baseline.append({"job_title": job["title"], "candidates": candidates})
    return {"method": "exact required-skill term coverage", "jobs": baseline}


def _render_keyword_baseline(baseline: dict[str, Any]) -> str:
    lines = [
        "# Deterministic keyword-match baseline",
        "",
        "Scores count exact required-skill terms present in extracted resume text, "
        "divided by the total required skills. This is not an LLM score or a "
        "measure of overall candidate suitability.",
        "",
    ]
    for job in baseline["jobs"]:
        lines.extend(
            [
                f"## {job['job_title']}",
                "",
                "| Rank | Resume | Keyword score |",
                "|---:|---|---:|",
            ]
        )
        for candidate in job["candidates"]:
            lines.append(
                f"| {candidate['rank']} | {candidate['resume']} | "
                f"{candidate['keyword_score']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank eight sample resumes for two jobs with two repeated LLM scores."
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Create a deterministic keyword baseline without making LLM calls.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        resumes = _load_resumes()
        jobs = _load_jobs()
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    if args.keyword_only:
        baseline = _run_keyword_baseline(resumes, jobs)
        output_path = TASK4_DIR / "keyword_baseline_v1.json"
        output_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path = TASK4_DIR / "keyword_baseline_v1.md"
        report_path.write_text(_render_keyword_baseline(baseline), encoding="utf-8")
        print(f"Keyword baseline written to {output_path} and {report_path}")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error(
            "OPENAI_API_KEY is not set; configure it locally to run 32 LLM comparisons."
        )
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def matcher(
        resume_name: str, resume_text: str, job: Job
    ) -> tuple[Match, dict[str, int | None]]:
        return _request_match(
            resume_name,
            resume_text,
            job,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
        )

    results = run_matchings(resumes, jobs, matcher)
    results["model"] = args.model
    json_path = TASK4_DIR / "results_v1.json"
    report_path = TASK4_DIR / "ranking_v1.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(_render_report(results, args.model), encoding="utf-8")
    print(f"Ranking data written to {json_path}")
    print(f"Ranking table written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
