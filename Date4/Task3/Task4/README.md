# Task 4: Job description matching

This directory includes two fictional job descriptions, a versioned matching
prompt, a repeatable LLM scoring runner, a deterministic keyword baseline, and
a four-line fairness note.

The runner reads the eight fictional Task1 resumes, sends each resume/job pair
to the configured model twice at temperature 0 (32 requests total), validates
the score and skill lists, logs token usage, computes score deltas, and ranks by
the mean of the two model scores. It writes `results_v1.json` and
`ranking_v1.md` only after a complete run. The report includes the keyword
coverage score beside both LLM scores.

Set `OPENAI_API_KEY` locally, then run:

```powershell
python Date4\Task3\Task4\match_resumes.py
```

By default the model is `gpt-4o-mini`; select another compatible model with
`OPENAI_MODEL` or `--model`. To create the deterministic keyword baseline
without a model call:

```powershell
python Date4\Task3\Task4\match_resumes.py --keyword-only
```

The LLM results are not a hiring recommendation. Review the fairness note and
the cited job/resume evidence before interpreting any score.

Run local tests with:

```powershell
python -m unittest discover -s Date4\Task3\Task4\tests -v
```
