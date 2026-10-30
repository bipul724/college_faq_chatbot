"""Load college documents (PDF/TXT/MD), split into chunks, store in Chroma."""
import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from rag.embeddings import get_embeddings

log = logging.getLogger(__name__)
COLLECTION = "college_faq"


def get_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_documents():
    docs = []
    for path in sorted(config.DOCS_DIR.rglob("*")):
        try:
            if path.suffix.lower() == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif path.suffix.lower() in {".txt", ".md"}:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            else:
                continue
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def build_index() -> dict:
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No .pdf/.txt/.md files found in {config.DOCS_DIR}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    store = get_store()
    try:
        store.delete_collection()
    except Exception:
        pass
    store = get_store()
    store.add_documents(chunks)
    return {"documents": len(docs), "chunks": len(chunks)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(build_index())
