"""Embedding backends. 'hf' = MiniLM (80MB, fine on 8GB RAM). 'hash' = offline fallback."""
import hashlib
import re
import numpy as np
from langchain_core.embeddings import Embeddings
import config


STOP = set("a an the is are was be to of for in on at by and or i my me can do does how what when where which who it this that with from as".split())


class HashEmbeddings(Embeddings):
    """Signed feature-hashing of words and bigrams. No download, good for tests/demo."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str):
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
        feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embeddings() -> Embeddings:
    if config.EMBEDDING_BACKEND == "hash":
        return HashEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
