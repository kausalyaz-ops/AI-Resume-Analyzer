# Task4: Search API and Evaluation

Install dependencies into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\Task4\requirements.txt
```

Set an API key before starting the server; protected endpoints expect it in
the `X-API-Key` header:

```powershell
$env:SEARCH_API_KEY = "replace-with-a-private-key"
.\.venv\Scripts\python.exe -m uvicorn Task4.search_api:app --reload
```

`GET /health` is public. `POST /search` accepts `query`, `top_k` (1-20), and
optional `doc_type` and `min_score`. `POST /documents` accepts a multipart
`file` upload in PDF, DOCX, or TXT format. The embedding model and starter
documents are loaded once when the application starts.

Run the 15-query chunk-size comparison:

```powershell
.\.venv\Scripts\python.exe .\Task4\evaluate_search.py
```

Run the six API tests:

```powershell
.\.venv\Scripts\python.exe -m pytest .\Task4\test_search_api.py -q
```
