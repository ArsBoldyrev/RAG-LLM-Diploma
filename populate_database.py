# populate_database.py
#test_commit
#test
import argparse
import os
import shutil
import time
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.errors import InternalError

from langchain_community.document_loaders import PyPDFLoader
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--batch-size", type=int, default=2000, help="Размер батча для upsert в Chroma.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Длина чанка при нарезке.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Перекрытие чанков.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Каталог с PDF.")
    args = parser.parse_args()

    if args.reset:
        print("-- Clearing Database --")
        clear_database()

    # 1) Загрузка документов (с прогрессом)
    t0 = time.perf_counter()
    documents = load_documents(args.data_path)
    t1 = time.perf_counter()

    # 2) Нарезка (с прогрессом)
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    t2 = time.perf_counter()

    # 3) Индексация (эмбеддинги + вставка) — прогресс уже есть
    stats = add_to_chroma(chunks, batch_size=args.batch_size)
    t3 = time.perf_counter()

    # ИТОГИ
    print("\n=================== STATS ===================")
    print(f"Исходных документов (страниц): {len(documents)}")
    print(f"Чанков всего:                  {len(chunks)}")
    print(f"Время загрузки файлов:        {(t1 - t0):.2f} c")
    print(f"Время нарезки на чанки:       {(t2 - t1):.2f} c")
    print(f"Время эмбеддингов (сумма):    {stats['time_embed']:.2f} c")
    print(f"Время вставки в Chroma:       {stats['time_upsert']:.2f} c")
    print(f"Итоговое время индексации:    {(t3 - t2):.2f} c")
    print(f"Новых чанков добавлено:       {stats['added_count']}")
    print(f"Фактический размер батча:     min={stats['min_batch_used']}, max={stats['max_batch_used']}")
    print("============================================\n")


def iter_pdf_files(root: str) -> List[str]:
    pdfs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, fn))
    return pdfs


def load_documents(root: str) -> List[Document]:
    """
    Грузим PDF по одному файлу, показывая прогресс-бар.
    Возвращаем список Document'ов (обычно это страницы PDF).
    """
    pdf_files = iter_pdf_files(root)
    documents: List[Document] = []
    for path in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        try:
            loader = PyPDFLoader(path)
            # у PyPDFLoader каждая страница — отдельный Document
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️  Пропускаю {path}: {e}")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 80) -> List[Document]:
    """
    Нарезаем постранично, чтобы показать прогресс.
    Результат идентичен split_documents(documents), но с tqdm.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Document] = []
    for doc in tqdm(documents, desc="Splitting to chunks", unit="doc"):
        # нарезаем по одному документу (странице)
        parts = splitter.split_documents([doc])
        chunks.extend(parts)
    return calculate_chunk_ids(chunks)


def add_to_chroma(chunks: List[Document], batch_size: int = 2000) -> dict:
    """
    Вставляет чанки батчами с авто-уменьшением размера при ошибке лимита.
    Показывает tqdm-прогресс по числу успешно добавленных чанков.
    Разделяет время на эмбеддинги и вставку.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    embedder = get_embedding_function()

    # Уже существующие ID (чтобы не дублировать)
    existing_items = db.get(include=[])  # IDs always included
    existing_ids = set(existing_items.get("ids", []))
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Только новые
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    if not new_chunks:
        print("✅ No new documents to add")
        return {"time_embed": 0.0, "time_upsert": 0.0, "added_count": 0, "min_batch_used": 0, "max_batch_used": 0}

    print(f"-- Adding new documents: {len(new_chunks)} (batch_size={batch_size}) --")

    time_embed_total = 0.0
    time_upsert_total = 0.0
    added_total = 0
    min_batch_used = 10**9
    max_batch_used = 0

    start = 0
    bs = batch_size
    n = len(new_chunks)

    pbar = tqdm(total=n, desc="Indexing (embed+upsert)", unit="chunks")
    try:
        while start < n:
            end = min(start + bs, n)
            part_docs = new_chunks[start:end]
            part_ids = [d.metadata["id"] for d in part_docs]
            part_texts = [d.page_content for d in part_docs]
            part_metas = [d.metadata for d in part_docs]

            # 1) эмбеддинги
            t0 = time.perf_counter()
            embeddings = embedder.embed_documents(part_texts)
            t1 = time.perf_counter()
            dt_embed = t1 - t0
            time_embed_total += dt_embed

            # 2) upsert (с авто-редукцией батча)
            while True:
                try:
                    t2 = time.perf_counter()
                    # низкоуровневый upsert — чтобы не пересчитывать эмбеддинги внутри Chroma
                    db._collection.upsert(
                        embeddings=embeddings,
                        metadatas=part_metas,
                        documents=part_texts,
                        ids=part_ids,
                    )
                    t3 = time.perf_counter()
                    dt_upsert = t3 - t2
                    time_upsert_total += dt_upsert

                    added = len(part_docs)
                    added_total += added
                    min_batch_used = min(min_batch_used, added)
                    max_batch_used = max(max_batch_used, added)

                    pbar.update(added)
                    pbar.set_postfix(bs=bs, embed_s=f"{dt_embed:.2f}", upsert_s=f"{dt_upsert:.2f}")
                    break

                except InternalError as e:
                    msg = str(e)
                    if "Batch size" in msg and "max batch size" in msg:
                        if bs <= 16:
                            raise
                        bs //= 2
                        pbar.set_postfix_str(f"reducing bs→{bs}")
                        # пересобираем окно под новый размер и пересчитываем эмбеддинги
                        end = min(start + bs, n)
                        part_docs = new_chunks[start:end]
                        part_ids = [d.metadata["id"] for d in part_docs]
                        part_texts = [d.page_content for d in part_docs]
                        part_metas = [d.metadata for d in part_docs]
                        t0 = time.perf_counter()
                        embeddings = embedder.embed_documents(part_texts)
                        t1 = time.perf_counter()
                        dt_embed = t1 - t0
                        time_embed_total += dt_embed
                        continue
                    else:
                        raise

            start = end
    finally:
        pbar.close()

    print("-- Done (persisted) --")
    return {
        "time_embed": time_embed_total,
        "time_upsert": time_upsert_total,
        "added_count": added_total,
        "min_batch_used": (0 if min_batch_used == 10**9 else min_batch_used),
        "max_batch_used": max_batch_used,
    }


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    ID вида "data/monopoly.pdf:6:2" (Источник:Страница:Индекс чанка).
    Оставляю твою логику без изменений.
    """
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

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
