import os
os.environ.update(EMBEDDING_BACKEND="hash", LLM_BACKEND="none", MIN_RELEVANCE="0.1",
                  DB_DIR="/tmp/faq_test_db")
import shutil
shutil.rmtree("/tmp/faq_test_db", ignore_errors=True)
from rag import ingest, qa
from app import app


def test_index_and_answers():
    stats = ingest.build_index()
    assert stats["chunks"] >= 3
    r = qa.answer("What is the minimum attendance required for exams?")
    assert "75%" in r["answer"] and r["sources"][0]["source"] == "attendance_and_exams.md"
    r = qa.answer("How many books can I borrow from the library?")
    assert "3 books" in r["answer"]
    assert qa.answer("")["sources"] == []


def test_api():
    c = app.test_client()
    assert c.get("/api/health").json["status"] == "ok"
    assert c.get("/").status_code == 200
    assert "answer" in c.post("/api/chat", json={"question": "hostel gate timing"}).json
