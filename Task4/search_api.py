import hashlib
import os
import re
import secrets
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK2_DIR = ROOT_DIR / "Task2"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = Field(default=None, min_length=1)
    min_score: float | None = Field(default=None, ge=-1, le=1)


class SearchResult(BaseModel):
    source: str
    page: int
    section: str
    score: float
    text: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class SearchEngine:
    def __init__(self, model=None):
        self.model = model
        self.chunks: dict[str, dict] = {}
        self.vectors = np.empty((0, VECTOR_SIZE), dtype=np.float32)
        self.ids: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model has not been loaded.")
        vectors = np.asarray(
            self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE}-dimensional embeddings.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def index_chunks(self, new_chunks: list[dict]) -> int:
        if not new_chunks:
            return 0
        vectors = self.embed([chunk["text"] for chunk in new_chunks])
        replaced_sources = {chunk["source"] for chunk in new_chunks}
        retained = [
            (chunk_id, self.chunks[chunk_id], self.vectors[index])
            for index, chunk_id in enumerate(self.ids)
            if self.chunks[chunk_id]["source"] not in replaced_sources
        ]

        updated_chunks = {chunk_id: chunk for chunk_id, chunk, _ in retained}
        updated_vectors = [vector for _, _, vector in retained]
        updated_ids = [chunk_id for chunk_id, _, _ in retained]
        for chunk, vector in zip(new_chunks, vectors):
            chunk_id = chunk["id"]
            updated_chunks[chunk_id] = chunk
            updated_vectors.append(vector)
            updated_ids.append(chunk_id)

        self.chunks = updated_chunks
        self.ids = updated_ids
        self.vectors = (
            np.asarray(updated_vectors, dtype=np.float32)
            if updated_vectors
            else np.empty((0, VECTOR_SIZE), dtype=np.float32)
        )
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int,
        doc_type: str | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        if not self.ids:
            return []
        query_vector = self.embed([query])
        scores = cosine_similarity(query_vector, self.vectors)[0]
        candidates = []
        normalized_type = doc_type.lower() if doc_type else None
        for index, chunk_id in enumerate(self.ids):
            chunk = self.chunks[chunk_id]
            if normalized_type and chunk["doc_type"].lower() != normalized_type:
                continue
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            candidates.append((score, chunk))
        candidates.sort(key=lambda result: result[0], reverse=True)
        return [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
                "score": score,
                "text": chunk["text"],
            }
            for score, chunk in candidates[:top_k]
        ]


def _stable_id(source: str, page: int, section: str, index: int, text: str) -> str:
    value = "\n".join((source, str(page), section, str(index), text))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_chunks(source: str, page: int, section: str, doc_type: str, text: str) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for index, chunk_text in enumerate(splitter.split_text(text)):
        chunks.append({
            "id": _stable_id(source, page, section, index, chunk_text),
            "source": source,
            "page": page,
            "section": section,
            "doc_type": doc_type,
            "text": chunk_text,
        })
    return chunks


def _default_chunks() -> list[dict]:
    if str(TASK2_DIR) not in sys.path:
        sys.path.insert(0, str(TASK2_DIR))
    import chunking_metadata_stable_ids as task2

    task2.create_sample_documents()
    documents = task2.extract_documents()
    chunks = []
    for document in documents:
        chunks.extend(_make_chunks(
            document["source"],
            document["page"],
            document["section"],
            document["doc_type"],
            document["text"],
        ))
    return chunks


def _extract_upload(filename: str, content: bytes) -> list[dict]:
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Supported upload types are PDF, DOCX, and TXT.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 20 MiB limit.")

    text_pages: list[tuple[int, str, str]] = []
    if extension == ".pdf":
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="PDF support requires PyMuPDF. Install Task4 requirements.",
                ) from exc
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    heading = re.search(r"^#+\s*(.+)$", text, re.MULTILINE)
                    text_pages.append((
                        page_number,
                        heading.group(1).strip() if heading else Path(filename).stem,
                        text,
                    ))
            pdf.close()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the PDF.") from exc
    elif extension == ".docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            section = Path(filename).stem
            paragraphs = []
            for paragraph in document.paragraphs:
                if paragraph.style.name.startswith("Heading"):
                    section = paragraph.text.strip() or section
                elif paragraph.text.strip():
                    paragraphs.append(paragraph.text.strip())
            text = "\n".join(paragraphs)
            if text:
                text_pages.append((1, section, text))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the DOCX.") from exc
    else:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="TXT uploads must be UTF-8 encoded.",
            ) from exc
        section = Path(filename).stem
        match = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
        if match:
            section = match.group(1).strip()
        if text:
            text_pages.append((1, section, text))

    chunks = []
    for page, section, text in text_pages:
        chunks.extend(_make_chunks(filename, page, section, extension[1:], text))
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in document.")
    return chunks


def create_app(engine: SearchEngine | None = None, api_key: str | None = None) -> FastAPI:
    injected_engine = engine
    configured_key = api_key if api_key is not None else os.getenv("SEARCH_API_KEY")
    search_engine = injected_engine or SearchEngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if search_engine.model is None:
            from sentence_transformers import SentenceTransformer

            search_engine.model = SentenceTransformer(MODEL_NAME)
            search_engine.index_chunks(_default_chunks())
        app.state.search_engine = search_engine
        yield

    app = FastAPI(title="Task4 Search API", version="1.0.0", lifespan=lifespan)

    def require_api_key(
        supplied_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if not configured_key:
            raise HTTPException(
                status_code=503,
                detail="Set SEARCH_API_KEY before using protected endpoints.",
            )
        if supplied_key is None or not secrets.compare_digest(supplied_key, configured_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/health")
    def health():
        return {"status": "ok", "indexed_chunks": len(search_engine.ids)}

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
    def search(request: SearchRequest):
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            doc_type=request.doc_type,
            min_score=request.min_score,
        )
        return {"results": results}

    @app.post("/documents", dependencies=[Depends(require_api_key)])
    async def upload_document(file: UploadFile = File(...)):
        source = Path(file.filename or "").name
        if not source or source in {".", ".."}:
            raise HTTPException(status_code=400, detail="A valid filename is required.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        chunks = _extract_upload(source, content)
        try:
            count = search_engine.index_chunks(chunks)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not embed and index upload.") from exc
        return {"source": source, "indexed_chunks": count}

    return app


app = create_app()
