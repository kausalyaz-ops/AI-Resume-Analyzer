import os
import statistics
import time
import uuid
from pathlib import Path

import faiss
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK2_DIR = ROOT_DIR / "Task2"
VECTOR_SIZE = 384


def explain_ann_and_hnsw():
    print("ANN and HNSW (5 lines):")
    print("1. Approximate Nearest Neighbor (ANN) search finds close vectors without checking every vector.")
    print("2. It trades a small amount of recall for much lower search latency on large datasets.")
    print("3. HNSW means Hierarchical Navigable Small World, a graph-based ANN index.")
    print("4. HNSW builds layered graphs so search can quickly navigate toward nearby vectors.")
    print("5. Its M, efConstruction, and efSearch settings balance memory, build time, speed, and recall.")


def median_search_time(index, queries, k, repeats=5):
    index.search(queries[:2], k)
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        index.search(queries, k)
        measurements.append(time.perf_counter() - started)
    return statistics.median(measurements)


def run_faiss_experiment():
    print("\nFAISS experiment (50,000 random vectors):")
    faiss.omp_set_num_threads(min(os.cpu_count() or 1, 8))
    vector_count = 50_000
    dimension = VECTOR_SIZE
    query_count = 100
    top_k = 10
    rng = np.random.default_rng(42)

    vectors = rng.standard_normal((vector_count, dimension), dtype=np.float32)
    queries = rng.standard_normal((query_count, dimension), dtype=np.float32)
    normalize(vectors, copy=False)
    normalize(queries, copy=False)

    exact_index = faiss.IndexFlatIP(dimension)
    exact_index.add(vectors)

    hnsw_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = 100
    hnsw_index.hnsw.efSearch = 512
    hnsw_index.add(vectors)

    exact_seconds = median_search_time(exact_index, queries, top_k)
    hnsw_seconds = median_search_time(hnsw_index, queries, top_k)
    exact_ids = exact_index.search(queries, top_k)[1]
    hnsw_ids = hnsw_index.search(queries, top_k)[1]
    recall = np.mean([
        len(set(exact_row).intersection(approx_row)) / top_k
        for exact_row, approx_row in zip(exact_ids, hnsw_ids)
    ])

    print(f"Vectors: {vector_count:,} x {dimension}; queries: {query_count}; k={top_k}")
    print(f"Exact search: {exact_seconds * 1000:.2f} ms")
    print(f"HNSW search:  {hnsw_seconds * 1000:.2f} ms")
    print(f"Recall@10: {recall:.3f}")


def load_task2_chunks():
    import sys

    if str(TASK2_DIR) not in sys.path:
        sys.path.insert(0, str(TASK2_DIR))
    import chunking_metadata_stable_ids as task2

    task2.create_sample_documents()
    documents = task2.extract_documents()
    return task2.make_chunks(documents, "fixed")


def qdrant_filter(*conditions):
    return models.Filter(must=list(conditions))


def point_id_for(stable_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_id))


def run_qdrant_demo():
    print("\nQdrant chunk collection (384-dimensional cosine vectors):")
    chunks = load_task2_chunks()
    if not chunks:
        raise RuntimeError("Task2 did not produce any chunks to index.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if vectors.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional embeddings, got {vectors.shape[1]}."
        )

    remote_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if remote_url:
        client = QdrantClient(url=remote_url, api_key=api_key)
        print("Using Qdrant server from QDRANT_URL.")
        is_local = False
    else:
        client = QdrantClient(":memory:")
        print("Using in-memory Qdrant for this run.")
        print(
            "Note: Qdrant local mode supports vector/payload-filter demos but "
            "does not build the server-side HNSW or payload indexes."
        )
        is_local = True

    collection_name = f"task3_chunks_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )

    before_index = client.get_collection(collection_name)
    print(
        "Index status before upsert: "
        f"status={before_index.status}, points={before_index.points_count}, "
        f"indexed_vectors={before_index.indexed_vectors_count}, "
        f"HNSW M={before_index.config.hnsw_config.m}"
    )

    point_structs = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk["section"],
            "created_date": chunk["created_date"],
            "chunk_index": chunk["chunk_index"],
            "doc_type": chunk["doc_type"].upper(),
            "text": chunk["text"],
            "stable_id": chunk["stable_id"],
        }
        point_structs.append(
            models.PointStruct(
                id=point_id_for(chunk["stable_id"]),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=point_structs, wait=True)
    after_index = client.get_collection(collection_name)
    print(
        "Index status after upsert:  "
        f"status={after_index.status}, points={after_index.points_count}, "
        f"indexed_vectors={after_index.indexed_vectors_count}, "
        f"HNSW M={after_index.config.hnsw_config.m}"
    )

    if not is_local:
        client.create_payload_index(
            collection_name,
            field_name="doc_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name,
            field_name="page",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Created Qdrant payload indexes for doc_type, page, and source.")

    query_vector = model.encode(
        ["Which documents discuss information relevant to this query?"],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()
    pdf_filter = qdrant_filter(
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="PDF"),
        )
    )
    pdf_results = client.query_points(
        collection_name,
        query=query_vector,
        query_filter=pdf_filter,
        limit=5,
        with_payload=True,
    ).points
    print(f"Filtered search (doc_type=PDF): {len(pdf_results)} results")
    for result in pdf_results:
        print(
            f"  {result.payload['source']} page {result.payload['page']} "
            f"(score {result.score:.3f})"
        )

    page_filter = qdrant_filter(
        models.FieldCondition(
            key="page",
            range=models.Range(gte=3),
        )
    )
    page_results = client.query_points(
        collection_name,
        query=query_vector,
        query_filter=page_filter,
        limit=5,
        with_payload=True,
    ).points
    print(f"Filtered search (page >= 3): {len(page_results)} results")

    count_first = client.count(collection_name, exact=True).count
    client.upsert(collection_name=collection_name, points=point_structs, wait=True)
    count_second = client.count(collection_name, exact=True).count
    print(
        f"Re-upsert: {count_first} points -> {count_second} points "
        f"({'no growth' if count_first == count_second else 'count changed'})"
    )

    source_to_delete = chunks[0]["source"]
    source_filter = qdrant_filter(
        models.FieldCondition(
            key="source",
            match=models.MatchValue(value=source_to_delete),
        )
    )
    count_source = client.count(
        collection_name,
        count_filter=source_filter,
        exact=True,
    ).count
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=source_filter),
        wait=True,
    )
    remaining = client.count(collection_name, exact=True).count
    print(
        f"Delete source filter ({source_to_delete}): "
        f"removed {count_source}; remaining points={remaining}"
    )
    client.close()


def print_database_comparison():
    print("\npgvector vs Qdrant:")
    rows = [
        ("Setup", "PostgreSQL extension + database", "Standalone service or embedded client"),
        ("Filtering", "SQL WHERE and relational joins", "Payload filters; payload indexes"),
        ("Scale", "Strong with PostgreSQL operations", "Purpose-built vector search; distributed options"),
        ("Choose when", "Vectors belong with relational data", "You need a dedicated vector database/API"),
    ]
    print(f"{'Aspect':<14} | {'pgvector':<42} | Qdrant")
    print("-" * 100)
    for aspect, pgvector, qdrant in rows:
        print(f"{aspect:<14} | {pgvector:<42} | {qdrant}")


def main():
    explain_ann_and_hnsw()
    run_faiss_experiment()
    run_qdrant_demo()
    print_database_comparison()
    print("\nConcepts: ANN, HNSW, pgvector, Qdrant, payload filters, upsert, deletion")


if __name__ == "__main__":
    main()
