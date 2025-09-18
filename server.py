# server.py
import os
import json
import time
from typing import List, Tuple, Generator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from get_embedding_function import get_embedding_function

# --- опциональный кросс-энкодерный реранкер ---
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

CHROMA_PATH = "chroma"
DATA_PATH = "data"

PROMPT_TEMPLATE = """
Ты — аккуратный ассистент. Отвечай ТОЛЬКО на основе «Контекста».
Если информации в контексте недостаточно — честно скажи, что её нет.
Форматируй ответ по-русски. Ставь квадратные ссылки на источники [1], [2] в местах фактов,
используя нумерацию из раздела «Источники».

Контекст:
{context}

---
Вопрос: {question}
"""

app = FastAPI(title="RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- utils ----------

def load_reranker(model_name: str = "BAAI/bge-reranker-base"):
    if not HAS_RERANKER:
        return None
    return CrossEncoder(model_name, trust_remote_code=True)

def rerank_pairs(cross_encoder, query: str, docs: List[str], top_k: int) -> List[Tuple[int, float]]:
    if cross_encoder is None or not docs:
        return [(i, 0.0) for i in range(min(top_k, len(docs)))]
    pairs = [(query, d) for d in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[: min(top_k, len(docs))]

def build_context(docs: List[Tuple[str, str]]) -> str:
    return "\n\n---\n\n".join(f"{label} {content}" for label, content in docs)

def format_sources(raw_docs) -> List[str]:
    rows = []
    for i, (doc, score) in enumerate(raw_docs, start=1):
        src = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0)
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        rows.append(f"[{i}] **{src}**, стр. {page}; `id={doc.metadata.get('id','?')}`\n> {snippet}")
    return rows

def sources_struct_with_snippets(results) -> List[dict]:
    """Подготовим структуру для фронта + дадим сниппеты для подсветки."""
    out = []
    for i, (doc, score) in enumerate(results, start=1):
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        out.append({
            "label": i,
            "file": doc.metadata.get("file_name", doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page", 0),
            "id": doc.metadata.get("id", "?"),
            "snippet": snippet
        })
    return out

def ensure_dirs():
    os.makedirs(DATA_PATH, exist_ok=True)

# ---------- routes ----------

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")  # НЕстриминговый фолбэк (оставляем)
def chat(
    question: str = Form(...),
    k: int = Form(5),
    fetch_k: int = Form(20),
    use_reranker: bool = Form(False),
    reranker_model: str = Form("BAAI/bge-reranker-base"),
    llm_model: str = Form("mistral"),
    temperature: float = Form(0.2),
):
    if not question.strip():
        return JSONResponse({"answer": "", "sources": []})

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    try:
        primary = db.max_marginal_relevance_search_with_score(question, k=min(k, fetch_k), fetch_k=fetch_k)
    except AttributeError:
        docs = db.max_marginal_relevance_search(question, k=min(k, fetch_k), fetch_k=fetch_k)
        primary = [(d, 0.0) for d in docs]

    if use_reranker and primary:
        ce = load_reranker(reranker_model)
        texts = [doc.page_content for (doc, _s) in primary]
        ranking = rerank_pairs(ce, question, texts, top_k=k)
        results = [(primary[idx][0], float(score)) for (idx, score) in ranking]
    else:
        results = primary[:k]

    if not results:
        return JSONResponse({"answer": "По контексту ничего не нашёл. Попробуй переформулировать запрос.", "sources": []})

    labeled_docs = [(f"[{i}]", doc.page_content) for i, (doc, _score) in enumerate(results, start=1)]
    context_text = build_context(labeled_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=question)

    llm = OllamaLLM(model=llm_model, temperature=temperature)
    answer = llm.invoke(prompt).strip()

    sources_md = "\n\n".join(format_sources(results))
    if sources_md:
        answer += "\n\n---\n**Источники**\n\n" + sources_md

    return JSONResponse({"answer": answer, "sources": sources_struct_with_snippets(results)})

# ========= НОВОЕ: SSE-СТРИМИНГ =========

def sse_pack(data: dict, event: str | None = None) -> str:
    """Упаковать объект в строку события SSE."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False)
    lines.append(f"data: {payload}")
    lines.append("")  # пустая строка = разделитель событий
    return "\n".join(lines)

@app.post("/api/chat/stream")
def chat_stream(
    question: str = Form(...),
    k: int = Form(5),
    fetch_k: int = Form(20),
    use_reranker: bool = Form(False),
    reranker_model: str = Form("BAAI/bge-reranker-base"),
    llm_model: str = Form("mistral"),
    temperature: float = Form(0.2),
):
    if not question.strip():
        def _empty():
            yield sse_pack({"message": ""}, event="done")
        return StreamingResponse(_gen(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
)       


    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    try:
        primary = db.max_marginal_relevance_search_with_score(question, k=min(k, fetch_k), fetch_k=fetch_k)
    except AttributeError:
        docs = db.max_marginal_relevance_search(question, k=min(k, fetch_k), fetch_k=fetch_k)
        primary = [(d, 0.0) for d in docs]

    if use_reranker and primary:
        ce = load_reranker(reranker_model)
        texts = [doc.page_content for (doc, _s) in primary]
        ranking = rerank_pairs(ce, question, texts, top_k=k)
        results = [(primary[idx][0], float(score)) for (idx, score) in ranking]
    else:
        results = primary[:k]

    if not results:
        def _nores():
            yield sse_pack({"message": "По контексту ничего не нашёл. Попробуй переформулировать запрос."}, event="token")
            yield sse_pack({}, event="done")
        return StreamingResponse(_nores(), media_type="text/event-stream",
                                 headers={
                                     "Cache-Control": "no-cache",
                                     "Connection": "keep-alive",
                                     "X-Accel-Buffering": "no",
                                 })

    labeled_docs = [(f"[{i}]", doc.page_content) for i, (doc, _score) in enumerate(results, start=1)]
    context_text = build_context(labeled_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=question)

    sources_payload = sources_struct_with_snippets(results)

    def _gen():
        # отправляем источники сразу, чтобы фронт что-то увидел мгновенно
        yield sse_pack({"sources": sources_payload}, event="sources")

        llm = OllamaLLM(model=llm_model, temperature=temperature)

        # Универсально читаем чанки (иногда это str, иногда объект с .content или .text)
        for chunk in llm.stream(prompt):
            try:
                if isinstance(chunk, str):
                    text = chunk
                elif hasattr(chunk, "content"):
                    text = chunk.content
                elif hasattr(chunk, "text"):
                    text = chunk.text
                else:
                    text = str(chunk)
            except Exception:
                text = str(chunk)
            if not text:
                continue
            yield sse_pack({"message": text}, event="token")

        yield sse_pack({}, event="done")

    return StreamingResponse(_gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
)

# ---------- upload & quick indexing ----------

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), chunk_size: int = Form(800), chunk_overlap: int = Form(80)):
    ensure_dirs()
    saved_paths = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            return JSONResponse({"error": f"Только PDF поддерживаются сейчас: {f.filename}"}, status_code=400)
        dest = os.path.join(DATA_PATH, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    added = quick_index_pdfs(saved_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return JSONResponse({"status": "ok", "indexed_chunks": added})

def quick_index_pdfs(paths: List[str], chunk_size=800, chunk_overlap=80) -> int:
    docs: List[Document] = []
    for p in paths:
        try:
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Пропускаю {p}: {e}")

    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    chunks = calculate_chunk_ids(chunks)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing = set(db.get(include=[]).get("ids", []))
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing]
    if not new_chunks:
        return 0

    bs = 1000
    start = 0
    total_added = 0
    while start < len(new_chunks):
        end = min(start + bs, len(new_chunks))
        part = new_chunks[start:end]
        try:
            db.add_documents(part, ids=[d.metadata["id"] for d in part])
            total_added += len(part)
            start = end
        except Exception:
            if bs <= 16:
                raise
            bs //= 2
            continue
    return total_added

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["file_name"] = os.path.basename(source) if source else "unknown"
        last_page_id = current_page_id
    return chunks
