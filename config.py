import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE = Path(__file__).parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE / "data" / "documents"))
DB_DIR = Path(os.getenv("DB_DIR", BASE / "data" / "chroma_db"))
COLLEGE_NAME = os.getenv("COLLEGE_NAME", "ABES Engineering College")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 500, 80, 4
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.25"))
