# Task 3: Validation and retry

`validate_resume.py` defines strict Pydantic v2 models for extracted resume
data. Experience is an integer from 0 through 50, and at least one skill with
non-empty evidence is required.

The runner parses each model response with `ResumeData.model_validate_json`.
Invalid JSON/data is sent back to the model as corrective feedback, up to three
attempts. Each attempt logs its status and the response's prompt, completion,
and total token counts. After three invalid responses it returns a structured
failure object with the last validation error and cumulative usage. An empty
resume returns a clear failure immediately, with zero attempts and zero tokens.

Run the local tests and simulated model retry:

```powershell
python -m pip install -r Date4\Task3\requirements.txt
python -m unittest discover -s Date4\Task3\tests -v
```

Run extraction against the included messy example after setting
`OPENAI_API_KEY` locally:

```powershell
python Date4\Task3\validate_resume.py
```

To check empty-input handling without an API call:

```powershell
python Date4\Task3\validate_resume.py Date4\Task3\samples\empty_resume.txt
```

The automated messy-resume test simulates an initial response with
`years_of_experience: 90`, followed by a corrected response with 5 years. It
checks the validation-feedback retry and per-attempt usage log. Tests do not
make external model requests.
