import json
import unittest
from pathlib import Path
import sys

TASK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK2_DIR))

from extract_resumes import validate_extraction  # noqa: E402


class ValidateExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resume = (
            "Skills\nPython, APIs, PostgreSQL\n"
            "Maintained Python utilities and wrote clear technical documentation."
        )
        self.valid = {
            "candidate_name": "Alex Rivera",
            "skills": [
                {"name": "Python", "evidence": "Python, APIs, PostgreSQL"},
                {
                    "name": "technical documentation",
                    "evidence": (
                        "Maintained Python utilities and wrote clear technical "
                        "documentation."
                    ),
                },
            ],
            "years_of_experience": None,
            "education": ["B.S. Computer Science"],
            "strengths": ["Maintains Python utilities"],
            "gaps": [],
        }

    def test_accepts_valid_schema_data_and_verbatim_skill_evidence(self) -> None:
        validate_extraction(self.valid, self.resume)

    def test_rejects_missing_or_additional_fields(self) -> None:
        for field in ("skills", "not_in_schema"):
            result = self.valid.copy()
            if field == "skills":
                del result[field]
            else:
                result[field] = "unexpected"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_extraction(result, self.resume)

    def test_rejects_evidence_not_found_in_resume(self) -> None:
        result = json.loads(json.dumps(self.valid))
        result["skills"][0]["evidence"] = "Python expert"
        with self.assertRaisesRegex(ValueError, "verbatim evidence"):
            validate_extraction(result, self.resume)

    def test_rejects_boolean_as_year_count(self) -> None:
        result = self.valid.copy()
        result["years_of_experience"] = True
        with self.assertRaisesRegex(ValueError, "non-negative number"):
            validate_extraction(result, self.resume)


if __name__ == "__main__":
    unittest.main()
