import os
import re
import io
from typing import List, Dict, Any

from fastapi.responses import Response

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


PDF_DIR = "pdfs"

# ---- Global state ----
doc_filenames: List[str] = []          # one per PDF
doc_pages: List[List[str]] = []        # list of pages per PDF
doc_fulltexts: List[str] = []          # full text per PDF (joined pages)
doc_sizes: List[int] = []              # size in bytes
doc_num_pages: List[int] = []          # number of pages per PDF

chunk_texts: List[str] = []            # one per page (for search)
chunk_meta: List[Dict[str, Any]] = []  # {doc_index, filename, page}

vectorizer = None
tfidf_matrix = None
embeddings = None
model = None

search_history: List[str] = []         # last 10 queries

app = FastAPI(title="PaperSense API")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # easier for local + future hosted frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Serve PDFs ------------------
if os.path.exists(PDF_DIR):
    app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")


# ------------------ Helpers ------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def extract_pages_from_pdf(path: str) -> List[str]:
    """Return a list of cleaned text, one string per page."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(clean_text(t))
    return pages


def load_pdfs():
    """Load all PDFs, store pages, fulltexts, sizes, metadata."""
    global doc_filenames, doc_pages, doc_fulltexts, doc_sizes, doc_num_pages

    doc_filenames = []
    doc_pages = []
    doc_fulltexts = []
    doc_sizes = []
    doc_num_pages = []

    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR, exist_ok=True)

    for fname in os.listdir(PDF_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(PDF_DIR, fname)
        print("Reading:", fname)

        pages = extract_pages_from_pdf(full_path)
        if not pages:
            continue

        doc_filenames.append(fname)
        doc_pages.append(pages)
        doc_fulltexts.append(" ".join(pages))
        doc_sizes.append(os.path.getsize(full_path))
        doc_num_pages.append(len(pages))


def build_indexes():
    """Build TF-IDF and embeddings over *pages* (page-level search)."""
    global chunk_texts, chunk_meta, vectorizer, tfidf_matrix, embeddings, model

    chunk_texts = []
    chunk_meta = []

    # Flatten pages into chunks
    for doc_idx, pages in enumerate(doc_pages):
        for page_idx, page_text in enumerate(pages):
            chunk_texts.append(page_text)
            chunk_meta.append(
                {
                    "doc_index": doc_idx,
                    "filename": doc_filenames[doc_idx],
                    "page": page_idx + 1,  # 1-based page number
                }
            )

    if not chunk_texts:
        raise RuntimeError("No pages found to index.")

    print("Building TF-IDF index over pages...")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings for pages...")
    embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)


# ------------------ Core search logic ------------------
def hybrid_search(query: str, top_k: int = 10):
    """Hybrid search over page-level chunks."""
    if vectorizer is None or tfidf_matrix is None or embeddings is None or model is None:
        raise RuntimeError("Search engine not initialized")

    tfidf_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]

    semantic_vec = model.encode([query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(semantic_vec, embeddings)[0]

    # Normalize each separately
    tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (
        semantic_scores.max() - semantic_scores.min() + 1e-8
    )

    combined = 0.4 * tfidf_norm + 0.6 * semantic_norm

    # Also normalize combined for score bucket
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    ranked = combined.argsort()[::-1]
    return ranked[:top_k], combined, combined_norm


def make_snippet(text: str, q: str, window=250):
    lower = text.lower()
    tokens = [t for t in q.lower().split() if len(t) > 2]

    pos = None
    for t in tokens:
        idx = lower.find(t)
        if idx != -1:
            pos = idx
            break

    start = 0 if pos is None else max(0, pos - window // 2)
    end = min(len(text), start + window)
    return text[start:end]


def highlight_html(snippet: str, q: str) -> str:
    """Return snippet with <mark>highlight</mark> instead of [word]."""
    highlighted = snippet
    for tok in q.split():
        if len(tok) < 3:
            continue
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
    return highlighted


def count_occurrences_in_doc(doc_index: int, query: str) -> int:
    """Count how many times tokens from query appear in the full PDF."""
    if doc_index < 0 or doc_index >= len(doc_fulltexts):
        return 0

    text = doc_fulltexts[doc_index].lower()
    tokens = [t for t in query.lower().split() if len(t) > 2]
    if not tokens:
        return 0

    total = 0
    for tok in tokens:
        total += len(re.findall(re.escape(tok), text))
    return total


def score_category(norm_score: float) -> str:
    """Return 'high', 'medium', 'low' based on normalized score (0–1)."""
    if norm_score >= 0.66:
        return "high"
    elif norm_score >= 0.33:
        return "medium"
    else:
        return "low"


# ------------------ Models ------------------
class FileMetadata(BaseModel):
    pages: int
    size_kb: float


class SearchRequest(BaseModel):
    query: str
    sort_by: str | None = "score"      # 'score', 'filename', 'occurrences'
    sort_order: str | None = "desc"    # 'asc' or 'desc'


class SearchResult(BaseModel):
    filename: str
    score: float
    snippet: str
    url: str
    page: int
    occurrences: int
    score_category: str
    metadata: FileMetadata


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    history: List[str] | None = None


class FileInfo(BaseModel):
    filename: str
    pages: int
    size_kb: float


# ------------------ Events ------------------
@app.on_event("startup")
def startup_event():
    print("Initializing PaperSense engine (page-level)...")
    load_pdfs()
    build_indexes()
    print("Ready!")


# ------------------ Endpoints ------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/files", response_model=List[FileInfo])
def list_files():
    """List all indexed PDFs with basic metadata."""
    infos: List[FileInfo] = []
    for i, fname in enumerate(doc_filenames):
        infos.append(
            FileInfo(
                filename=fname,
                pages=doc_num_pages[i],
                size_kb=round(doc_sizes[i] / 1024.0, 1),
            )
        )
    return infos


@app.get("/history")
def get_history():
    """Return the last 10 search queries (most recent first)."""
    return {"history": list(reversed(search_history))}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    # Track history (keep last 10)
    search_history.append(q)
    if len(search_history) > 10:
        search_history.pop(0)

    ranked_indices, scores, norm_scores = hybrid_search(q)
    results: List[SearchResult] = []

    for idx in ranked_indices[:10]:  # top 10 chunks/pages
        meta = chunk_meta[idx]
        doc_idx = meta["doc_index"]
        filename = meta["filename"]
        page_num = meta["page"]

        snippet_raw = make_snippet(chunk_texts[idx], q)
        snippet_hl = highlight_html(snippet_raw, q)

        occ = count_occurrences_in_doc(doc_idx, q)
        cat = score_category(norm_scores[idx])

        meta_obj = FileMetadata(
            pages=doc_num_pages[doc_idx],
            size_kb=round(doc_sizes[doc_idx] / 1024.0, 1),
        )

        results.append(
            SearchResult(
                filename=filename,
                score=float(round(float(scores[idx]), 4)),
                snippet=snippet_hl,
                url=f"/pdfs/{filename}",
                page=page_num,
                occurrences=occ,
                score_category=cat,
                metadata=meta_obj,
            )
        )

    # Sorting on backend (optional)
    sort_by = (req.sort_by or "score").lower()
    sort_order = (req.sort_order or "desc").lower()
    reverse = sort_order == "desc"

    if sort_by == "filename":
        results.sort(key=lambda r: r.filename.lower(), reverse=reverse)
    elif sort_by == "occurrences":
        results.sort(key=lambda r: r.occurrences, reverse=reverse)
    else:  # default: score
        results.sort(key=lambda r: r.score, reverse=reverse)

    return SearchResponse(
        results=results,
        total_results=len(results),
        history=list(reversed(search_history)),
    )


@app.post("/reload")
def reload():
    load_pdfs()
    build_indexes()
    return {"status": "reloaded", "documents": len(doc_filenames)}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload a new PDF, save it, and rebuild indexes."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    os.makedirs(PDF_DIR, exist_ok=True)

    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    load_pdfs()
    build_indexes()

    return {"status": "uploaded", "filename": file.filename}


@app.get("/thumbnail/{filename}")
def thumbnail(filename: str):
    """
    Return a PNG thumbnail of the first page of the given PDF.
    Requires 'pymupdf' (install with: pip install pymupdf).
    """
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF (fitz) is not installed. Install with 'pip install pymupdf'.",
        )

    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise HTTPException(status_code=400, detail="PDF has no pages")

    page = doc.load_page(0)
    zoom = 1.5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img_bytes = pix.tobytes("png")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

import fitz  # pymupdf

@app.get("/thumbnail/{filename}")
def thumbnail(filename: str):
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high-res thumbnail
        image_bytes = pix.tobytes("png")

        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
