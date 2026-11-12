"""Vector retrieval. Nguyên tắc: index once, retrieve many (embedding được lưu cache)."""
import hashlib
import json

import numpy as np

from config import EMBED_BACKEND, PROC, SBERT_MODEL, TOP_K


class Retriever:
    def __init__(self) -> None:
        with open(PROC / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        texts = [c["text"] for c in self.chunks]
        self.backend = EMBED_BACKEND
        if self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(SBERT_MODEL)
            self.vectors = self._cached_embeddings(texts)
        else:  # tfidf: dự phòng khi không tải được model, dùng được cả để so sánh
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            self.vectors = self.vectorizer.fit_transform(texts)

    def _cached_embeddings(self, texts):
        digest = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
        path = PROC / f"embeddings_{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        np.save(path, vectors)
        return vectors

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if self.backend == "sbert":
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.vectors @ q                     # đã chuẩn hóa nên tích vô hướng = cosine
        else:
            q = self.vectorizer.transform([query])
            scores = (self.vectors @ q.T).toarray().ravel()   # TF-IDF đã chuẩn hóa L2
        best = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]
