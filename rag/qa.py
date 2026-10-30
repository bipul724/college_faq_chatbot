"""Retrieval + answer generation (LangChain LCEL: prompt | llm | parser)."""
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import config
from rag.ingest import get_store

log = logging.getLogger(__name__)
NO_ANSWER = ("I couldn't find that in the college documents. "
             "Please contact the college office or check the official website.")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the FAQ assistant for {college}. Answer ONLY using the context. "
               "If the context does not contain the answer, say you don't know and suggest "
               "contacting the college office. Be short and clear."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def retrieve(question: str):
    hits = get_store().similarity_search_with_score(question, k=config.TOP_K)
    out = []
    for doc, dist in hits:
        score = 1 - dist  # cosine distance -> similarity
        if score >= config.MIN_RELEVANCE:
            out.append((doc, round(float(score), 3)))
    return out


def _generate(question: str, hits):
    if config.LLM_BACKEND != "ollama":
        return None
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_URL,
                         temperature=0.1, num_ctx=2048)
        chain = PROMPT | llm | StrOutputParser()
        context = "\n\n".join(d.page_content for d, _ in hits)
        return chain.invoke({"college": config.COLLEGE_NAME, "context": context,
                             "question": question}).strip()
    except Exception as exc:  # Ollama not running etc.
        log.warning("LLM unavailable, using extractive fallback: %s", exc)
        return None


def answer(question: str) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "Please type a question.", "sources": [], "mode": "none"}
    hits = retrieve(question)
    if not hits:
        return {"answer": NO_ANSWER, "sources": [], "mode": "none"}
    text = _generate(question, hits)
    mode = "llm"
    if not text:
        mode = "extractive"
        text = "Here is the most relevant information I found:\n\n" + hits[0][0].page_content.strip()
    sources = [{"source": d.metadata.get("source", "?"), "page": d.metadata.get("page"),
                "score": s, "snippet": d.page_content.strip()[:200]} for d, s in hits]
    return {"answer": text, "sources": sources, "mode": mode}
