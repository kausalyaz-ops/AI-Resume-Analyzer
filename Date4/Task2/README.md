# Task 2: Extraction prompt and schema

## Version 1 artifacts

- [`schema_v1.json`](./schema_v1.json) defines the six required top-level
  extraction fields. Each skill is an object containing a name and verbatim
  evidence from the resume.
- [`system_prompt_v1.md`](./system_prompt_v1.md) defines the role, task, output
  format, and non-invention constraints.
- [`extract_resumes.py`](./extract_resumes.py) calls the OpenAI Chat Completions
  API for each of the eight fictional Task1 resumes with strict JSON Schema
  response mode and `temperature: 0`. It validates the returned structure and
  checks each skill quote against the resume text.

The prompt intentionally sets `years_of_experience` to `null` unless a total is
explicitly stated. Year-only job dates do not provide exact employment
durations. Since no target job description is supplied, `gaps` should remain
empty unless a resume explicitly identifies a missing qualification.

## Run the extraction

Set `OPENAI_API_KEY` in the local environment, then run from any directory:

```powershell
python Date4\Task2\extract_resumes.py
```

By default the script uses `gpt-4o-mini` and saves the eight model responses to
`results_v1.json`. Set `OPENAI_MODEL` or pass `--model` to select a compatible
model. Do not put credentials in the repository.

The request uses the OpenAI `json_schema` response format in strict mode; it
fails explicitly if the key is missing, a call fails, returned JSON is invalid,
or skill evidence is not copied from the source resume.

## Manual verification

Manual verification of three model responses and an error count must be filled
in after the extraction run. Record each checked field/evidence quote against
its source resume; count an error for each incorrect, unsupported, or omitted
fact. The run is not considered verified until this review is recorded.

Run local contract tests with:

```powershell
python -m unittest discover -s Date4\Task2\tests -v
```
