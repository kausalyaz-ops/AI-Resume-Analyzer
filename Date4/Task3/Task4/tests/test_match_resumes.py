import unittest
from pathlib import Path
import sys

TASK4_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK4_DIR))

from match_resumes import (  # noqa: E402
    keyword_match_score,
    run_matchings,
    validate_match,
)


class MatchResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = {
            "job_id": "sample",
            "title": "Sample",
            "description": "Example description",
            "required_skills": ["Python", "SQL"],
        }

    def test_validates_match_and_score_range(self) -> None:
        validate_match(
            {
                "score": 50,
                "matched_skills": ["Python"],
                "missing_skills": ["SQL"],
                "reasoning": "Python is evidenced; SQL is not mentioned.",
            },
            self.job,
        )
        with self.assertRaisesRegex(ValueError, "0 through 100"):
            validate_match(
                {
                    "score": 101,
                    "matched_skills": [],
                    "missing_skills": ["Python", "SQL"],
                    "reasoning": "No skills are evidenced.",
                },
                self.job,
            )

    def test_keyword_score_is_exact_term_coverage(self) -> None:
        self.assertEqual(keyword_match_score("Python developer using SQL", ["Python", "SQL"]), 100)
        self.assertEqual(keyword_match_score("Pythonic workflows", ["Python", "SQL"]), 0)
        self.assertEqual(keyword_match_score("", []), 0)

    def test_each_resume_job_pair_runs_twice_and_ranks_stably(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_match(resume_name: str, _text: str, job: dict[str, object]):
            calls.append((resume_name, str(job["job_id"])))
            required = job["required_skills"]
            matched = ["Python"] if "Python" in required else []
            missing = [skill for skill in required if skill not in matched]
            return (
                {
                    "score": 70,
                    "matched_skills": matched,
                    "missing_skills": missing,
                    "reasoning": "The result is based on the provided job requirements.",
                },
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        jobs = [
            self.job,
            {
                **self.job,
                "job_id": "other",
                "required_skills": ["Python"],
            },
        ]
        resumes = [("resume_01.pdf", "Python"), ("resume_02.pdf", "SQL")]
        results = run_matchings(resumes, jobs, fake_match)

        self.assertEqual(len(calls), 8)
        self.assertEqual([candidate["rank"] for candidate in results["jobs"][0]["candidates"]], [1, 2])
        self.assertTrue(all(candidate["consistent"] for candidate in results["jobs"][0]["candidates"]))
        self.assertEqual(results["jobs"][0]["candidates"][0]["runs"][0]["score"], 70)


if __name__ == "__main__":
    unittest.main()
