import json
import logging
import unittest
from pathlib import Path
import sys

TASK3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK3_DIR))

from validate_resume import (  # noqa: E402
    FailureResult,
    ModelResponse,
    ResumeData,
    TokenUsage,
    validate_with_retries,
)


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> ModelResponse:
        self.messages.append([message.copy() for message in messages])
        content = next(self.responses)
        return ModelResponse(
            content=content,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )


def _resume_json(years: int) -> str:
    return json.dumps(
        {
            "candidate_name": "Casey Morrow",
            "skills": [
                {"name": "customer support", "evidence": "Skills: customer support"}
            ],
            "years_of_experience": years,
            "education": ["Certificate in Customer Care"],
            "strengths": ["Created clearer handoff notes"],
            "gaps": [],
        }
    )


class ValidateResumeTests(unittest.TestCase):
    def test_messy_resume_retries_with_validation_feedback_then_succeeds(self) -> None:
        resume = (
            TASK3_DIR / "samples" / "messy_resume.txt"
        ).read_text(encoding="utf-8")
        model = FakeModel([_resume_json(90), _resume_json(5)])
        with self.assertLogs("resume_validation", level=logging.INFO) as logs:
            result = validate_with_retries(resume, model)

        self.assertIsInstance(result, ResumeData)
        self.assertEqual(result.years_of_experience, 5)
        self.assertEqual(len(model.messages), 2)
        self.assertIn("less than or equal to 50", model.messages[1][-1]["content"])
        self.assertIn(
            "Attempt 1: validation error (years_of_experience:",
            logs.output[0],
        )
        self.assertIn("Attempt 2: success", logs.output[1])
        self.assertIn("prompt_tokens=20", logs.output[0])
        self.assertIn("completion_tokens=10", logs.output[1])

    def test_empty_resume_returns_failure_without_requesting_model(self) -> None:
        empty_resume = (TASK3_DIR / "samples" / "empty_resume.txt").read_text(
            encoding="utf-8"
        )
        model = FakeModel([])

        result = validate_with_retries(empty_resume, model)

        self.assertIsInstance(result, FailureResult)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 0)
        self.assertIn("Resume is empty", result.error)
        self.assertEqual(model.messages, [])

    def test_three_invalid_responses_return_clear_failure_and_sum_usage(self) -> None:
        model = FakeModel([_resume_json(90), _resume_json(90), _resume_json(90)])

        result = validate_with_retries("Non-empty resume text", model)

        self.assertIsInstance(result, FailureResult)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)
        self.assertIn("failed after 3 attempts", result.error)
        self.assertEqual(result.token_usage.total_tokens, 90)
        self.assertEqual(len(model.messages), 3)

    def test_year_constraints_and_non_empty_skills(self) -> None:
        for years in (-1, 51):
            with self.subTest(years=years), self.assertRaises(ValueError):
                ResumeData.model_validate_json(_resume_json(years))

        empty_skills = json.loads(_resume_json(5))
        empty_skills["skills"] = []
        with self.assertRaises(ValueError):
            ResumeData.model_validate_json(json.dumps(empty_skills))

    def test_json_schema_requires_nullable_years_field(self) -> None:
        schema = ResumeData.model_json_schema()

        self.assertIn("years_of_experience", schema["required"])
        years_schema = schema["properties"]["years_of_experience"]
        self.assertIn(
            {"type": "null"},
            years_schema["anyOf"],
        )


if __name__ == "__main__":
    unittest.main()
