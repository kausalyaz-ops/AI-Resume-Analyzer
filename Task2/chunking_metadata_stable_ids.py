import hashlib
import re
from datetime import date
from itertools import islice
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required. Install it with: pip install -r Task2/requirements.txt"
        ) from exc
import numpy as np
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


TASK2_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = TASK2_DIR / "sample_documents"
CREATED_DATE = "2026-11-18"


def create_sample_documents():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    documents = [
        ("climate.pdf", "Climate and Weather", [
            "Climate describes long-term atmospheric patterns over many years.",
            "A warm ocean current can transport heat across a large region.",
            "Flood risk rises when heavy rainfall exceeds drainage capacity.",
        ]),
        ("space.pdf", "Space Exploration", [
            "Telescopes collect light that reveals distant galaxies and planets.",
            "An orbit is the repeated path of an object around another body.",
            "Solar panels convert sunlight into electrical power for spacecraft.",
        ]),
        ("medicine.docx", "Medicine and Health", [
            "Vaccines train the immune system to recognize dangerous pathogens.",
            "Preventive screening can identify disease before symptoms appear.",
            "A clinical trial compares an intervention with a control treatment.",
        ]),
        ("cooking.txt", "Cooking and Nutrition", [
            "Cooking vegetables with gentle heat helps preserve their texture.",
            "Balanced meals combine protein, carbohydrates, and healthy fats.",
            "A recipe should list ingredients and the order of preparation.",
        ]),
        ("technology.txt", "Computer Technology", [
            "A database stores structured information for efficient retrieval.",
            "Encryption protects sensitive data from unauthorized reading.",
            "Machine learning finds patterns in examples supplied during training.",
        ]),
    ]

    for filename, heading, paragraphs in documents:
        path = SAMPLES_DIR / filename
        if path.suffix == ".pdf":
            pdf = fitz.open()
            for page_number in range(3):
                page = pdf.new_page()
                page.insert_text((72, 72), f"{heading} - Page {page_number + 1}")
                lines = []
                for repetition in range(12):
                    for paragraph_index, paragraph in enumerate(paragraphs):
                        text = (
                            f"Section {chr(65 + paragraph_index)}: "
                            f"{paragraph} Additional context for page {page_number + 1}, "
                            f"part {repetition + 1}, explains this topic in practical detail."
                        )
                        lines.append(text)
                page.insert_textbox(
                    (72, 110, 540, 760),
                    "\n".join(lines),
                    fontsize=7,
                )
            pdf.save(path)
            pdf.close()
        elif path.suffix == ".docx":
            word_document = Document()
            word_document.add_heading(heading, level=1)
            for paragraph_index, paragraph in enumerate(paragraphs):
                word_document.add_heading(f"Section {chr(65 + paragraph_index)}", level=2)
                word_document.add_paragraph(paragraph)
            word_document.save(path)
        else:
            with path.open("w", encoding="utf-8") as text_file:
                text_file.write(f"# {heading}\n")
                for paragraph_index, paragraph in enumerate(paragraphs):
                    text_file.write(f"## Section {chr(65 + paragraph_index)}\n")
                    text_file.write(paragraph + "\n\n")


def extract_documents():
    extracted = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            pdf = fitz.open(path)
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                heading_match = re.search(r"(.+?) - Page", text)
                heading = heading_match.group(1).strip() if heading_match else "Unspecified"
                extracted.append({
                    "source": path.name,
                    "page": page_number,
                    "section": heading,
                    "doc_type": "pdf",
                    "text": text,
                })
            pdf.close()
        elif path.suffix.lower() == ".docx":
            word_document = Document(path)
            heading = "Medicine and Health"
            section = "Section A"
            parts = []
            for paragraph in word_document.paragraphs:
                if paragraph.style.name.startswith("Heading"):
                    heading = paragraph.text if paragraph.style.name == "Heading 1" else section
                    section = paragraph.text if paragraph.style.name == "Heading 2" else section
                elif paragraph.text.strip():
                    parts.append(f"{heading}\n{section}\n{paragraph.text.strip()}")
            extracted.append({
                "source": path.name,
                "page": 1,
                "section": "Medicine and Health",
                "doc_type": "docx",
                "text": "\n\n".join(parts),
            })
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            heading_match = re.search(r"# ([^\n]+)", text)
            heading = heading_match.group(1).strip() if heading_match else "Unspecified"
            extracted.append({
                "source": path.name,
                "page": 1,
                "section": heading,
                "doc_type": "txt",
                "text": text,
            })
    return extracted


def whitespace_tokens(text):
    return re.findall(r"\S+", text)


def fixed_size_chunking(text, size=512, overlap=50):
    tokens = whitespace_tokens(text)
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(tokens):
        chunks.append(" ".join(tokens[start:start + size]))
        start += step
    return chunks


def sentence_chunking(text, target_tokens=512):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    current_size = 0
    for sentence in sentences:
        sentence_size = len(whitespace_tokens(sentence))
        if current and current_size + sentence_size > target_tokens:
            chunks.append(" ".join(current))
            current = []
            current_size = 0
        current.append(sentence)
        current_size += sentence_size
    if current:
        chunks.append(" ".join(current))
    return chunks


def stable_id(chunk):
    canonical = "\n".join([
        chunk["source"],
        str(chunk["page"]),
        chunk["section"],
        chunk["doc_type"],
        chunk["text"],
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_metadata(document, text_chunks):
    chunks = []
    for chunk_index, text in enumerate(text_chunks):
        chunk = {
            "source": document["source"],
            "page": document["page"],
            "section": document["section"],
            "created_date": CREATED_DATE,
            "chunk_index": chunk_index,
            "doc_type": document["doc_type"],
            "text": text,
        }
        chunk["stable_id"] = stable_id(chunk)
        chunks.append(chunk)
    return chunks


def describe_chunk(chunk):
    return (
        f"{chunk['source']} | {chunk['doc_type']} | "
        f"page {chunk['page']} | section: {chunk['section']} | "
        f"chunk {chunk['chunk_index']} | created {chunk['created_date']} "
        f"| id {chunk['stable_id'][:12]}"
    )


def sentence_was_cut(first_chunk, next_chunk):
    first_words = whitespace_tokens(first_chunk)
    next_words = whitespace_tokens(next_chunk)
    if len(first_words) < 12 or len(next_words) < 12:
        return None
    return (
        f'"{" ".join(first_words[-12:])} [CUT] '
        f'{" ".join(next_words[:12])}"'
    )


def make_chunks(documents, strategy):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for document in documents:
        if strategy == "fixed":
            text_chunks = fixed_size_chunking(document["text"], 512, 50)
        elif strategy == "sentence":
            text_chunks = sentence_chunking(document["text"], 512)
        else:
            text_chunks = splitter.split_text(document["text"])
        chunks.extend(attach_metadata(document, text_chunks))
    return chunks


def choose_chunk_size(documents, model):
    questions = [
        ("What is climate?", "climate.pdf", 1),
        ("How do ocean currents affect heat?", "climate.pdf", 1),
        ("What can a telescope reveal?", "space.pdf", 1),
        ("How does a spacecraft get electricity?", "space.pdf", 2),
        ("How do vaccines work?", "medicine.docx", 1),
        ("What is a clinical trial?", "medicine.docx", 1),
        ("How can meals contain balanced nutrients?", "cooking.txt", 1),
        ("What is a recipe?", "cooking.txt", 1),
        ("What is a database used for?", "technology.txt", 1),
        ("How does encryption protect data?", "technology.txt", 1),
    ]
    print("\nChunk-size hit-rate experiment (10 known-page questions):")
    print("Size | Hits | Hit rate")

    for size in (128, 256, 512, 1024):
        hits = 0
        for question, expected_source, expected_page in questions:
            candidates = []
            for document in documents:
                for chunk in fixed_size_chunking(document["text"], size, 50):
                    candidates.append((document, chunk))
            if not candidates:
                continue
            candidate_embeddings = model.encode(
                [chunk for _, chunk in candidates],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            question_embedding = model.encode([question], convert_to_numpy=True)
            scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
            best_document, _ = candidates[int(np.argmax(scores))]
            if (
                best_document["source"] == expected_source
                and best_document["page"] == expected_page
            ):
                hits += 1
        print(f"{size:4d} | {hits:4d} | {hits / len(questions):.0%}")


def incremental_index(chunks):
    return {chunk["stable_id"]: chunk for chunk in chunks}


def main():
    create_sample_documents()
    documents = extract_documents()
    print(f"Extracted {len(documents)} documents:")
    for document in documents:
        print(
            f"  {document['source']} | {document['doc_type']} | "
            f"{document['page']} page(s) | {len(whitespace_tokens(document['text']))} tokens"
        )

    fixed_chunks = make_chunks(documents, "fixed")
    sentence_chunks = make_chunks(documents, "sentence")
    recursive_chunks = make_chunks(documents, "recursive")
    print("\nChunk count comparison:")
    print(f"Fixed 512/50: {len(fixed_chunks)} chunks")
    print(f"Sentence-based: {len(sentence_chunks)} chunks")
    print(f"Recursive 512/50: {len(recursive_chunks)} chunks")

    cut_example = None
    for document in documents:
        document_fixed_chunks = fixed_size_chunking(document["text"], 512, 50)
        for index in range(len(document_fixed_chunks) - 1):
            cut_example = sentence_was_cut(
                document_fixed_chunks[index],
                document_fixed_chunks[index + 1],
            )
            if cut_example:
                break
        if cut_example:
            break
    print("\nSentence cut in half example:")
    print(cut_example if cut_example else "No cut detected in the current sample.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    choose_chunk_size(documents, model)

    print("\nMetadata example:")
    print(describe_chunk(fixed_chunks[0]))

    index_run_1 = incremental_index(fixed_chunks)
    index_run_2 = incremental_index(fixed_chunks)
    print("\nIncremental re-indexing:")
    print(f"Run 1: {len(index_run_1)} points")
    print(f"Run 2: {len(index_run_2)} points")

    changed_documents = [dict(document) for document in documents]
    for document in changed_documents:
        if document["source"] == "technology.txt":
            document["text"] = document["text"].replace(
                "Encryption protects sensitive data",
                "Strong encryption safeguards private data",
            )
    changed_chunks = make_chunks(changed_documents, "fixed")
    old_ids = {chunk["stable_id"] for chunk in fixed_chunks}
    new_ids = {chunk["stable_id"] for chunk in changed_chunks}
    replaced = len(old_ids - new_ids)
    index_run_3 = incremental_index(changed_chunks)
    print(
        f"Run 3 (1 edit): {len(index_run_3)} points, "
        f"{replaced} replaced"
    )

    print("\nConcepts: chunking strategies, metadata, stable ids, incremental re-indexing")


if __name__ == "__main__":
    main()
