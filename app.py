# app.py
import os
from typing import List, Tuple

import gradio as gr
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

# --- опциональный кросс-энкодерный реранкер ---
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ты — аккуратный ассистент. Отвечай ТОЛЬКО на основе «Контекста».
Если информации в контексте недостаточно — честно скажи, что её нет.

Форматируй ответ по-русски. Ставь квадратные ссылки на источники [1], [2] в местах фактов,
используя нумерацию из раздела «Источники» ниже.

Контекст:
{context}

---
Вопрос: {question}
"""

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

def rag_answer(
    question: str,
    chat_history: List[Tuple[str, str]],
    k: int = 5,
    fetch_k: int = 20,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base",
    llm_model: str = "mistral",
    temperature: float = 0.2,
):
    if not question or not question.strip():
        return chat_history, gr.update(value="")

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 1) первичный ретрив (MMR) — кросс-версионная совместимость
    try:
        primary = db.max_marginal_relevance_search_with_score(
            question, k=min(k, fetch_k), fetch_k=fetch_k
        )
    except AttributeError:
        # старые версии возвращают без score
        docs = db.max_marginal_relevance_search(
            question, k=min(k, fetch_k), fetch_k=fetch_k
        )
        primary = [(d, 0.0) for d in docs]

    # 2) реранк (опционально)
    if use_reranker and primary:
        ce = load_reranker(reranker_model)
        texts = [doc.page_content for (doc, _s) in primary]
        ranking = rerank_pairs(ce, question, texts, top_k=k)
        results = [(primary[idx][0], float(score)) for (idx, score) in ranking]
    else:
        results = primary[:k]

    if not results:
        chat_history = chat_history + [(question, "По контексту ничего не нашёл. Попробуй переформулировать запрос.")]
        return chat_history, ""

    # 3) контекст и промпт
    labeled_docs = [(f"[{i}]", doc.page_content) for i, (doc, _score) in enumerate(results, start=1)]
    context_text = build_context(labeled_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=question
    )

    # 4) генерация
    llm = OllamaLLM(model=llm_model, temperature=temperature)
    answer = llm.invoke(prompt).strip()

    # 5) источники (markdown)
    sources_md = "\n\n".join(format_sources(results))
    if sources_md:
        answer += "\n\n---\n**Источники**\n\n" + sources_md

    chat_history = chat_history + [(question, answer)]
    return chat_history, ""

# ---------- UI ----------
CUSTOM_CSS = """
.gradio-container {max-width: 920px !important; margin: 0 auto;}
.message.user {background: #0ea5e9; color: white;}
.message.assistant {background: #111827; color: #e5e7eb;}
.message {border-radius: 10px; padding: 12px 14px;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="RAG Chat") as demo:
    gr.Markdown("## 🔎 RAG Chat — локально на Ollama + Chroma\nСтиль ответа как в ChatGPT, с источниками.")

    with gr.Row():
        with gr.Column(scale=3):
            k = gr.Slider(1, 10, value=5, step=1, label="Top-k (в LLM)")
            fetch_k = gr.Slider(5, 50, value=20, step=1, label="Fetch-k (MMR pool)")
            use_reranker = gr.Checkbox(False, label="Включить реранкер (CrossEncoder)")
            reranker_model = gr.Textbox(value="BAAI/bge-reranker-base", label="Модель реранкера", visible=True)
            llm_model = gr.Textbox(value="mistral", label="LLM (Ollama модель)")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Температура")

        with gr.Column(scale=7):
            # ВАЖНО: тип по умолчанию (список пар), чтобы соответствовать нашему state
            chat = gr.Chatbot(height=520)
            user_input = gr.Textbox(placeholder="Задай вопрос по загруженным PDF…", label="Сообщение")
            with gr.Row():
                send_btn = gr.Button("Отправить", variant="primary")
                clear_btn = gr.Button("Очистить диалог")

    state = gr.State([])

    def on_send(user_msg, history, k, fetch_k, use_reranker, reranker_model, llm_model, temperature):
        return rag_answer(
            question=user_msg,
            chat_history=history,
            k=int(k),
            fetch_k=int(fetch_k),
            use_reranker=bool(use_reranker),
            reranker_model=reranker_model,
            llm_model=llm_model,
            temperature=float(temperature),
        )

    send_btn.click(
        on_send,
        inputs=[user_input, state, k, fetch_k, use_reranker, reranker_model, llm_model, temperature],
        outputs=[chat, user_input],
    )
    user_input.submit(
        on_send,
        inputs=[user_input, state, k, fetch_k, use_reranker, reranker_model, llm_model, temperature],
        outputs=[chat, user_input],
    )
    clear_btn.click(lambda: ([], ""), outputs=[state, user_input])

if __name__ == "__main__":
    # Можно пробросить OLLAMA_BASE_URL, если не localhost:11434
    # export OLLAMA_BASE_URL=http://localhost:11434
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.getenv("PORT", 7860)),
                ) 
