import logging
from flask import Flask, jsonify, render_template, request
import config
from rag import ingest, qa

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

STUDENT = {"name": "Bipul Chamoli", "roll": "2300320100069",
           "branch": "CSE", "college": "ABES Engineering College"}


@app.get("/")
def index():
    return render_template("index.html", college=config.COLLEGE_NAME, student=STUDENT)


@app.post("/api/chat")
def chat():
    q = (request.get_json(silent=True) or {}).get("question", "")
    if len(q) > 500:
        return jsonify(error="Question is too long (max 500 characters)."), 400
    try:
        return jsonify(qa.answer(q))
    except Exception as exc:
        app.logger.exception("chat failed")
        return jsonify(error="Index not ready. Run: python -m rag.ingest"), 500


@app.post("/api/reindex")
def reindex():
    try:
        return jsonify(ingest.build_index())
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.get("/api/health")
def health():
    return jsonify(status="ok", llm=config.LLM_BACKEND, embeddings=config.EMBEDDING_BACKEND)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
