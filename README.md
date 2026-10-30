# LLM-Based FAQ Chatbot for College Website (P_110)

**Domain:** LLM · Education / Cross-Industry AI Tooling
**Student:** Bipul Chamoli · Roll No. 2300320100069 · CSE · ABES Engineering College
**Hardware target:** Laptop with 8 GB RAM · **Stack:** LangChain, Chroma (vector DB), Flask

## What it does
Answers student FAQs (attendance, exams, fees, hostel, library, placements) by retrieving
passages from college documents (PDF/TXT/MD) and generating a grounded answer.
If nothing relevant is found it says so instead of guessing.

## Architecture (RAG)
```
documents -> loader -> text splitter (500/80) -> embeddings -> Chroma (cosine)
question  -> embed -> top-4 retrieval -> relevance filter -> prompt | LLM -> answer + sources
                                                       \-> (no LLM) extractive fallback
```
| Part | Choice | Why (8 GB laptop) |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | ~80 MB, runs on CPU |
| Vector DB | Chroma, persisted on disk | no server needed |
| LLM | Ollama `llama3.2:3b` (~2 GB) | fits in 8 GB, fully local/private |
| Web | Flask + plain HTML/JS | light |

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # Windows: copy .env.example .env
# Optional but recommended for generated answers:
#   install Ollama from https://ollama.com, then:  ollama pull llama3.2:3b
python -m rag.ingest        # build the vector index
python app.py               # open http://localhost:5000
```
No Ollama? Set `LLM_BACKEND=none` and the bot returns the best matching passage with sources.
No internet for the embedding model? Set `EMBEDDING_BACKEND=hash` (lower quality, works offline).

## Using real college data
Delete the SAMPLE files in `data/documents/`, add the real PDFs/notes (handbook, fee circulars,
exam rules), then run `python -m rag.ingest` again (or POST `/api/reindex`).
The included documents are **sample placeholders, not official ABES policies**.

## Tests
`pytest -q` (uses the offline hash embeddings, no downloads).

## Project layout
`app.py` Flask routes · `config.py` settings · `rag/ingest.py` indexing · `rag/qa.py` retrieval+LLM ·
`rag/embeddings.py` embedding backends · `templates/` `static/` UI · `tests/` pytest

## Future work
Conversation memory, Hindi support, admin upload page, feedback buttons, evaluation set (precision@k).
