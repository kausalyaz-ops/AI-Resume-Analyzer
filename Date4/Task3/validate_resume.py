"""Validate model-extracted resume JSON and retry with validation feedback."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError


TASK3_DIR = Path(__file__).resolve().parent
PROMPT_PATH = TASK3_DIR.parent / "Task2" / "system_prompt_v1.md"
API_URL = "https://api.openai.com/v1/chat/completions"
MAX_ATTEMPTS = 3
LOGGER = logging.getLogger("resume_validation")


class SkillEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_name: str | None
    skills: list[SkillEvidence] = Field(min_length=1)
    years_of_experience: int | None = Field(ge=0, le=50)
    education: list[str]
    strengths: list[str]
    gaps: list[str]


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: TokenUsage


@dataclass(frozen=True)
class FailureResult:
    success: bool
    error: str
    attempts: int
    token_usage: TokenUsage


ModelRequester = Callable[[list[dict[str, str]]], ModelResponse]


class ModelRequestError(RuntimeError):
    """Raised when a model request cannot produce a usable response."""


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "response"
        details.append(f"{location}: {issue['msg']}")
    return "; ".join(details)


def _accumulate_usage(total: TokenUsage, current: TokenUsage) -> TokenUsage:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    values: dict[str, int | None] = {}
    for field in fields:
        old_value = getattr(total, field)
        new_value = getattr(current, field)
        values[field] = (
            old_value + new_value
            if old_value is not None and new_value is not None
            else old_value if new_value is None else new_value
        )
    return TokenUsage(**values)


def validate_with_retries(
    resume_text: str,
    request_model: ModelRequester,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> ResumeData | FailureResult:
    """Request and validate JSON, retrying invalid output with its error."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not resume_text.strip():
        return FailureResult(
            success=False,
            error="Resume is empty; extraction was not attempted.",
            attempts=0,
            token_usage=TokenUsage(0, 0, 0),
        )

    messages = [
        {
            "role": "system",
            "content": PROMPT_PATH.read_text(encoding="utf-8"),
        },
        {
            "role": "user",
            "content": f"Extract structured data from this resume:\n\n{resume_text}",
        },
    ]
    total_usage = TokenUsage(0, 0, 0)
    last_error = "No valid response received."

    for attempt in range(1, max_attempts + 1):
        response: ModelResponse | None = None
        try:
            response = request_model(messages)
            total_usage = _accumulate_usage(total_usage, response.usage)
            validated = ResumeData.model_validate_json(response.content)
        except ValidationError as error:
            last_error = _format_validation_error(error)
            LOGGER.info(
                "Attempt %d: validation error (%s); prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s",
                attempt,
                last_error,
                response.usage.prompt_tokens if response else None,
                response.usage.completion_tokens if response else None,
                response.usage.total_tokens if response else None,
            )
            if attempt < max_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON failed Pydantic validation: "
                                f"{last_error}. Return a corrected JSON object "
                                "matching the schema. Do not invent facts."
                            ),
                        },
                    ]
                )
            continue
        except ModelRequestError as error:
            last_error = str(error)
            LOGGER.info(
                "Attempt %d: request error (%s); prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s",
                attempt,
                last_error,
                response.usage.prompt_tokens if response else None,
                response.usage.completion_tokens if response else None,
                response.usage.total_tokens if response else None,
            )
            if attempt < max_attempts:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous model request failed: "
                            f"{last_error}. Please return a corrected JSON response."
                        ),
                    }
                )
            continue

        LOGGER.info(
            "Attempt %d: success; prompt_tokens=%s completion_tokens=%s "
            "total_tokens=%s",
            attempt,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )
        return validated

    return FailureResult(
        success=False,
        error=f"Validation failed after {max_attempts} attempts: {last_error}",
        attempts=max_attempts,
        token_usage=total_usage,
    )


def _request_openai(
    messages: list[dict[str, str]], *, api_key: str, model: str
) -> ModelResponse:
    request_body = {
        "model": model,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "resume_data",
                "strict": True,
                "schema": ResumeData.model_json_schema(),
            },
        },
        "messages": messages,
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
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise ModelRequestError(f"HTTP {error.code}") from error
    except (URLError, TimeoutError) as error:
        raise ModelRequestError(f"Network request failed: {error}") from error
    except json.JSONDecodeError as error:
        raise ModelRequestError("Model endpoint returned invalid JSON") from error

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelRequestError("Model endpoint response is missing message content") from error
    if not isinstance(content, str):
        raise ModelRequestError("Model endpoint message content is not text")

    usage_body = body.get("usage") or {}
    usage = TokenUsage(
        prompt_tokens=usage_body.get("prompt_tokens"),
        completion_tokens=usage_body.get("completion_tokens"),
        total_tokens=usage_body.get("total_tokens"),
    )
    return ModelResponse(content=content, usage=usage)


def _result_dict(result: ResumeData | FailureResult) -> dict[str, object]:
    if isinstance(result, FailureResult):
        return asdict(result)
    return {"success": True, "data": result.model_dump(mode="json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract one resume and validate the model response with retries."
    )
    parser.add_argument(
        "resume",
        nargs="?",
        type=Path,
        default=TASK3_DIR / "samples" / "messy_resume.txt",
        help="Resume text file (defaults to the included messy sample).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the result JSON.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model (default: OPENAI_MODEL or gpt-4o-mini).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )
    try:
        resume_text = args.resume.read_text(encoding="utf-8-sig")
    except OSError as error:
        parser.error(f"Cannot read resume {args.resume}: {error}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not resume_text.strip():
        result: ResumeData | FailureResult = validate_with_retries(
            resume_text, lambda _messages: (_ for _ in ()).throw(
                AssertionError("Empty resumes must not call the model")
            )
        )
    elif not api_key:
        parser.error("OPENAI_API_KEY is not set; configure it in the local environment.")
    else:
        result = validate_with_retries(
            resume_text,
            lambda messages: _request_openai(
                messages, api_key=api_key, model=args.model
            ),
        )

    serialized = json.dumps(_result_dict(result), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if isinstance(result, ResumeData) else 1


if __name__ == "__main__":
    raise SystemExit(main())
