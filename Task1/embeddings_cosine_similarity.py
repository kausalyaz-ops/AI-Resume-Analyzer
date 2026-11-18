import json
import os
import time
import urllib.request
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity(a, b):
    """Calculate cosine similarity without using a cosine-similarity library."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same dimensions")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def embed_with_api(sentences):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "SKIPPED (set OPENAI_API_KEY to call the embedding API)"

    model = os.getenv("EMBEDDING_API_MODEL", "text-embedding-3-small")
    payload = json.dumps({"model": model, "input": sentences}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started

    ordered = sorted(result["data"], key=lambda item: item["index"])
    embeddings = np.asarray([item["embedding"] for item in ordered], dtype=float)
    return elapsed, embeddings


sentences = [
    "The feline is resting on the couch.",
    "A cat sleeps peacefully on the sofa.",
    "Python is a popular language for data science.",
    "Code written in Python helps scientists analyze information.",
    "Heavy rain flooded several streets in the city.",
    "The storm caused water to cover downtown roads.",
    "The physician prescribed antibiotics for the infection.",
    "A doctor gave medication to treat an illness.",
    "Fresh pasta with tomato sauce is served for dinner.",
    "The local team won the football match at the stadium.",
]

print("Cosine similarity tests:")
test_pairs = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0]),
]
for a, b in test_pairs:
    custom = cosine_similarity(a, b)
    sklearn = float(
        sklearn_cosine_similarity(np.asarray(a).reshape(1, -1),
                                  np.asarray(b).reshape(1, -1))[0][0]
    )
    print(f"{a} vs {b} -> {custom:.1f}")
    print(f"  scikit-learn comparison: {sklearn:.1f}")


print("\nEmbedding 10 sentences:")
model = SentenceTransformer("all-MiniLM-L6-v2")

started = time.perf_counter()
local_embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
local_time = time.perf_counter() - started
print(f"Local model dimension: {local_embeddings.shape[1]}")
print(f"Local model time: {local_time:.3f} seconds")

api_result = embed_with_api(sentences)
if isinstance(api_result[0], float):
    api_time, api_embeddings = api_result
    print(f"Embedding API dimension: {api_embeddings.shape[1]}")
    print(f"Embedding API time: {api_time:.3f} seconds")
else:
    print(f"Embedding API result: {api_result[1]}")
    api_embeddings = None


matrix = sklearn_cosine_similarity(local_embeddings)
print("\nSimilarity matrix:")
print(np.round(matrix, 3))

ranked_pairs = []
for i, j in combinations(range(len(sentences)), 2):
    ranked_pairs.append((float(matrix[i, j]), i, j))
ranked_pairs.sort(reverse=True)

print("\n3 most similar sentence pairs:")
for rank, (score, i, j) in enumerate(ranked_pairs[:3], start=1):
    print(f"{rank}. {score:.3f} | {sentences[i]} <> {sentences[j]}")

print("\nNo-common-word high-similarity pair:")
no_common_pairs = []
for score, i, j in ranked_pairs:
    words_i = set(sentences[i].lower().replace(".", "").split())
    words_j = set(sentences[j].lower().replace(".", "").split())
    if not words_i.intersection(words_j):
        no_common_pairs.append((score, i, j))

if no_common_pairs:
    score, i, j = no_common_pairs[0]
    print(f"{score:.3f} | {sentences[i]} <> {sentences[j]}")
else:
    print("No pair met the no-common-word condition.")


print("\nUse case 1 - Semantic search:")
search_documents = [
    "A recipe explains how to bake bread.",
    "Python code is useful for machine learning.",
    "The football team scored three goals.",
    "A physician helps patients recover from illness.",
    "The train carried travelers across the country.",
    "Rain clouds covered the city all morning.",
]
search_queries = [
    "How can I prepare homemade bread?",
    "Which language should I use for AI projects?",
    "What happened in the soccer match?",
]
document_embeddings = model.encode(search_documents, convert_to_numpy=True)
query_embeddings = model.encode(search_queries, convert_to_numpy=True)
search_scores = sklearn_cosine_similarity(query_embeddings, document_embeddings)
for query, scores in zip(search_queries, search_scores):
    best = int(np.argmax(scores))
    print(f"{query} -> {search_documents[best]} ({scores[best]:.3f})")


print("\nUse case 2 - Duplicate detection:")
duplicate_items = [
    "How do I reset my account password?",
    "I cannot sign in after forgetting my password.",
    "Resetting a password for my login.",
    "The train arrived at the station on time.",
    "The train reached the station punctually.",
    "Please update your delivery address.",
]
duplicate_threshold = 0.75
duplicate_embeddings = model.encode(duplicate_items, convert_to_numpy=True)
duplicate_matrix = sklearn_cosine_similarity(duplicate_embeddings)
duplicate_found = False
for i, j in combinations(range(len(duplicate_items)), 2):
    if duplicate_matrix[i, j] >= duplicate_threshold:
        duplicate_found = True
        print(
            f"{duplicate_matrix[i, j]:.3f} (threshold {duplicate_threshold:.2f}) | "
            f"{duplicate_items[i]} <> {duplicate_items[j]}"
        )
if not duplicate_found:
    print(f"No duplicates found at threshold {duplicate_threshold:.2f}.")


print("\nUse case 3 - K-Means clustering:")
cluster_documents = [
    "The soccer team won the championship.",
    "The tennis player won the final set.",
    "A runner finished the marathon race.",
    "Python helps with artificial intelligence.",
    "A laptop computer uses software.",
    "A software system uses machine learning.",
    "Fresh apples and bread are at the market.",
    "Pasta and tomato sauce are served for dinner.",
    "A chef prepares meals in the kitchen.",
]
cluster_embeddings = model.encode(cluster_documents, convert_to_numpy=True)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(cluster_embeddings)

candidate_names = {
    "sports": {"soccer", "team", "football", "tennis", "player", "match", "championship"},
    "technology": {"python", "computer", "software", "artificial", "intelligence", "code"},
    "food": {"fresh", "bread", "market", "apples", "pasta", "food"},
    "health": {"doctor", "doctors", "patient", "medicine", "disease", "treatment", "illness"},
}

for cluster_id in range(3):
    members = [text for text, label in zip(cluster_documents, labels) if label == cluster_id]
    words = set(" ".join(members).lower().replace(".", "").split())
    cluster_name = max(
        candidate_names,
        key=lambda name: len(words.intersection(candidate_names[name]))
        or -1,
    )
    print(f"{cluster_name.capitalize()} cluster:")
    for member in members:
        print(f"  - {member}")


print("\nLocal vs API model choice:")
print("Local embeddings avoid API fees and keep sentence data on your own machine.")
print("Local models also work offline but require model downloads and compute resources.")
print("API embeddings are convenient for consistent managed capacity and large-scale workloads.")
print("Use an API for low setup cost; use local embeddings for privacy, control, and offline use.")
