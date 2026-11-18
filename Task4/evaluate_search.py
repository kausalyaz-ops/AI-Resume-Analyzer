import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search_api import ROOT_DIR, TASK2_DIR, MODEL_NAME, SearchEngine


if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))
import chunking_metadata_stable_ids as task2


EVALUATION_QUERIES = [
    ("What long-term patterns describe the climate?", "climate.pdf", 1),
    ("How does a warm ocean current move heat?", "climate.pdf", 1),
    ("What can cause city streets to flood?", "climate.pdf", 1),
    ("Which instrument collects light from faraway galaxies?", "space.pdf", 1),
    ("What path does an object follow around another body?", "space.pdf", 1),
    ("How do spacecraft generate electrical power?", "space.pdf", 1),
    ("How can vaccines help the immune system?", "medicine.docx", 1),
    ("Why is preventive screening useful?", "medicine.docx", 1),
    ("How does a clinical trial compare treatments?", "medicine.docx", 1),
    ("How can gentle heat preserve vegetables?", "cooking.txt", 1),
    ("What nutrients make a balanced meal?", "cooking.txt", 1),
    ("What information should a recipe list?", "cooking.txt", 1),
    ("How does a database store information?", "technology.txt", 1),
    ("What protects sensitive data from being read?", "technology.txt", 1),
    ("How does machine learning find patterns?", "technology.txt", 1),
]


def chunks_for_size(documents: list[dict], chunk_size: int) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(50, chunk_size - 1),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["text"])):
            chunk = {
                "source": document["source"],
                "page": document["page"],
                "section": document["section"],
                "doc_type": document["doc_type"],
                "text": text,
            }
            chunk["id"] = task2.stable_id({
                **chunk,
                "chunk_index": index,
            })
            chunks.append(chunk)
    return chunks


def evaluate(model, documents: list[dict], chunk_size: int) -> dict:
    chunks = chunks_for_size(documents, chunk_size)
    chunk_vectors = np.asarray(
        model.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    query_vectors = np.asarray(
        model.encode(
            [query for query, _, _ in EVALUATION_QUERIES],
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    scores = cosine_similarity(query_vectors, chunk_vectors)
    reciprocal_ranks = []
    hits = {1: 0, 3: 0, 5: 0}

    for query_index, (_, expected_source, expected_page) in enumerate(EVALUATION_QUERIES):
        order = np.argsort(scores[query_index])[::-1]
        relevant_ranks = [
            rank
            for rank, chunk_index in enumerate(order, start=1)
            if (
                chunks[chunk_index]["source"] == expected_source
                and chunks[chunk_index]["page"] == expected_page
            )
        ]
        first_rank = relevant_ranks[0] if relevant_ranks else None
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        for cutoff in hits:
            if first_rank is not None and first_rank <= cutoff:
                hits[cutoff] += 1

    query_count = len(EVALUATION_QUERIES)
    return {
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "hit_at_1": hits[1] / query_count,
        "hit_at_3": hits[3] / query_count,
        "hit_at_5": hits[5] / query_count,
        "mrr": sum(reciprocal_ranks) / query_count,
    }


def main():
    task2.create_sample_documents()
    documents = task2.extract_documents()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    results = [
        evaluate(model, documents, chunk_size)
        for chunk_size in (256, 512)
    ]

    print(f"Retrieval evaluation: {len(EVALUATION_QUERIES)} queries, judged by source and page")
    print("Config | Chunk size | Chunks | hit@1 | hit@3 | hit@5 | MRR")
    for label, result in zip(("A", "B"), results):
        print(
            f"{label}     | {result['chunk_size']:10d} | {result['chunk_count']:6d} | "
            f"{result['hit_at_1']:.2f} | {result['hit_at_3']:.2f} | "
            f"{result['hit_at_5']:.2f} | {result['mrr']:.2f}"
        )

    better = max(
        results,
        key=lambda result: (result["hit_at_3"], result["mrr"], result["hit_at_5"]),
    )
    label = "A" if better is results[0] else "B"
    print(
        f"Decision: Config {label} (chunk size {better['chunk_size']}) is better "
        f"by the priority hit@3, then MRR. "
        f"hit@3={better['hit_at_3']:.2f}, MRR={better['mrr']:.2f}."
    )


if __name__ == "__main__":
    main()
