from fastapi.testclient import TestClient

from search_api import SearchEngine, create_app


class TinyModel:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            for word in text.lower().split():
                vector[sum(word.encode("utf-8")) % len(vector)] += 1.0
            vectors.append(vector)
        return vectors


def make_client():
    engine = SearchEngine(TinyModel())
    engine.index_chunks([
        {
            "id": "climate-1",
            "source": "climate.pdf",
            "page": 3,
            "section": "Weather",
            "doc_type": "pdf",
            "text": "Climate describes long-term weather patterns and changing temperatures.",
        },
        {
            "id": "cooking-1",
            "source": "cooking.txt",
            "page": 1,
            "section": "Food",
            "doc_type": "txt",
            "text": "A recipe explains how to cook vegetables and prepare food.",
        },
    ])
    app = create_app(engine=engine, api_key="test-key")
    return TestClient(app)


def test_health_is_public():
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["indexed_chunks"] == 2


def test_search_requires_api_key():
    with make_client() as client:
        response = client.post("/search", json={"query": "weather"})
    assert response.status_code == 401


def test_search_returns_citation_fields():
    with make_client() as client:
        response = client.post(
            "/search",
            headers={"X-API-Key": "test-key"},
            json={"query": "climate weather", "top_k": 2},
        )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert {"source", "page", "section", "score", "text"} <= result.keys()
    assert result["source"] == "climate.pdf"


def test_top_k_validation_rejects_out_of_range_values():
    with make_client() as client:
        headers = {"X-API-Key": "test-key"}
        too_small = client.post("/search", headers=headers, json={"query": "x", "top_k": 0})
        too_large = client.post("/search", headers=headers, json={"query": "x", "top_k": 21})
    assert too_small.status_code == 422
    assert too_large.status_code == 422


def test_search_filters_by_type_and_minimum_score():
    with make_client() as client:
        response = client.post(
            "/search",
            headers={"X-API-Key": "test-key"},
            json={
                "query": "climate weather",
                "top_k": 20,
                "doc_type": "PDF",
                "min_score": 0.0,
            },
        )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert all(result["source"].endswith(".pdf") for result in results)
    assert all(result["score"] >= 0.0 for result in results)


def test_upload_indexes_and_replaces_same_source():
    with make_client() as client:
        headers = {"X-API-Key": "test-key"}
        first = client.post(
            "/documents",
            headers=headers,
            files={"file": ("notes.txt", b"## Notes\nA unique observatory studies stars.", "text/plain")},
        )
        count_after_first = client.get("/health").json()["indexed_chunks"]
        second = client.post(
            "/documents",
            headers=headers,
            files={"file": ("notes.txt", b"## Notes\nA unique observatory studies planets.", "text/plain")},
        )
        count_after_second = client.get("/health").json()["indexed_chunks"]
        unsupported = client.post(
            "/documents",
            headers=headers,
            files={"file": ("notes.csv", b"a,b", "text/csv")},
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert count_after_first == 3
    assert count_after_second == count_after_first
    assert unsupported.status_code == 415
