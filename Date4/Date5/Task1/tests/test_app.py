import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app.app)
        self.valid_result = app.AnalyzeResponse(
            candidate_name="Alex Example",
            skills=[{"name": "Python", "evidence": "Skills: Python"}],
            years_of_experience=None,
            education=[],
            strengths=["Python development"],
            gaps=[],
            score=78,
            matched_skills=["Python"],
            missing_skills=["SQL"],
            reasoning="Python is evidenced; SQL is not mentioned.",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
            },
            estimated_cost={
                "model": "gpt-4o-mini",
                "input_usd_per_1m_tokens": 0.15,
                "output_usd_per_1m_tokens": 0.6,
                "estimated_usd": 0.00003,
            },
        )

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("app._analyze_with_openai")
    def test_analyze_accepts_resume_text_and_returns_usage_and_cost(
        self, analyze_mock
    ) -> None:
        analyze_mock.return_value = self.valid_result

        response = self.client.post(
            "/analyze",
            data={
                "resume_text": "Skills: Python",
                "job_description": "Python Developer role",
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["candidate_name"], "Alex Example")
        self.assertEqual(result["score"], 78)
        self.assertEqual(result["token_usage"]["total_tokens"], 125)
        self.assertEqual(result["estimated_cost"]["estimated_usd"], 0.00003)
        analyze_mock.assert_called_once_with("Skills: Python", "Python Developer role")

    @patch("app._analyze_with_openai")
    def test_analyze_accepts_txt_upload(self, analyze_mock) -> None:
        analyze_mock.return_value = self.valid_result
        response = self.client.post(
            "/analyze",
            data={"job_description": "Python role"},
            files={"resume_file": ("resume.txt", b"Skills: Python", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        analyze_mock.assert_called_once_with("Skills: Python", "Python role")

    @patch("app._analyze_with_openai")
    def test_analyze_accepts_pdf_upload(self, analyze_mock) -> None:
        analyze_mock.return_value = self.valid_result
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Skills: Python")
        pdf_bytes = document.tobytes()
        document.close()

        response = self.client.post(
            "/analyze",
            data={"job_description": "Python role"},
            files={"resume_file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        analyze_mock.assert_called_once_with("Skills: Python", "Python role")

    def test_rejects_scanned_pdf_with_clear_message(self) -> None:
        document = pymupdf.open()
        document.new_page()
        pdf_bytes = document.tobytes()
        document.close()
        response = self.client.post(
            "/analyze",
            data={"job_description": "Python role"},
            files={"resume_file": ("scan.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("No selectable text", response.json()["detail"])

    @patch("app.urlopen")
    def test_upstream_key_is_never_returned_and_cost_uses_configured_rates(
        self, urlopen_mock
    ) -> None:
        upstream_data = {
            "candidate_name": "Alex Example",
            "skills": [{"name": "Python", "evidence": "Skills: Python"}],
            "years_of_experience": None,
            "education": [],
            "strengths": [],
            "gaps": [],
            "score": 78,
            "matched_skills": ["Python"],
            "missing_skills": ["SQL"],
            "reasoning": "Python is evidenced.",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(upstream_data),
                                    "refusal": None,
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 1_000_000,
                            "total_tokens": 2_000_000,
                        },
                    }
                ).encode("utf-8")

        urlopen_mock.return_value = FakeResponse()
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key-not-for-output",
                "OPENAI_MODEL": "gpt-4o-mini",
                "OPENAI_INPUT_USD_PER_1M_TOKENS": "0.2",
                "OPENAI_OUTPUT_USD_PER_1M_TOKENS": "0.8",
            },
            clear=True,
        ):
            result = app._analyze_with_openai("Skills: Python", "Python role")

        self.assertEqual(result.estimated_cost.estimated_usd, 1.0)
        self.assertNotIn("test-key-not-for-output", result.model_dump_json())
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-key-not-for-output"
        )

    @patch("app.urlopen")
    def test_rejects_skill_evidence_not_in_resume(self, urlopen_mock) -> None:
        upstream_data = {
            "candidate_name": "Alex Example",
            "skills": [{"name": "Python", "evidence": "Python expert"}],
            "years_of_experience": None,
            "education": [],
            "strengths": [],
            "gaps": [],
            "score": 78,
            "matched_skills": ["Python"],
            "missing_skills": [],
            "reasoning": "Python is evidenced.",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(upstream_data)}}
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ).encode("utf-8")

        urlopen_mock.return_value = FakeResponse()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with self.assertRaises(app.HTTPException) as raised:
                app._analyze_with_openai("Skills: Python", "Python role")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("evidence not found", raised.exception.detail)

    def test_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/analyze",
            data={"job_description": "Python role"},
            files={"resume_file": ("resume.docx", b"resume", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_rejects_resume_over_size_limit(self) -> None:
        response = self.client.post(
            "/analyze",
            data={"job_description": "Python role"},
            files={
                "resume_file": (
                    "resume.txt",
                    io.BytesIO(b"x" * (app.MAX_RESUME_BYTES + 1)),
                    "text/plain",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    def test_requires_exactly_one_resume_input(self) -> None:
        both = self.client.post(
            "/analyze",
            data={
                "resume_text": "resume",
                "job_description": "role",
            },
            files={"resume_file": ("resume.txt", b"resume", "text/plain")},
        )
        neither = self.client.post(
            "/analyze", data={"job_description": "role"}
        )

        self.assertEqual(both.status_code, 422)
        self.assertEqual(neither.status_code, 422)

    def test_openapi_exposes_docs_and_expected_request_fields(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertIn("/analyze", schema["paths"])
        self.assertIn("/health", schema["paths"])
        self.assertEqual(
            schema["paths"]["/analyze"]["post"]["requestBody"]["content"].keys(),
            {"multipart/form-data"},
        )


if __name__ == "__main__":
    unittest.main()
