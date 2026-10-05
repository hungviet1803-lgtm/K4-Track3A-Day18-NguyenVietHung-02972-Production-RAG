from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re, unicodedata
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép") nhưng query người dùng gõ
    "nghỉ phép" → phải replace("_", " ") để token của doc và query khớp nhau.
    Lợi ích còn lại của bước này: tách dấu câu khỏi từ ("ngày." → "ngày .").
    """
    from underthesea import word_tokenize

    text = unicodedata.normalize("NFC", text)
    if not text.strip():
        return ""
    segmented = word_tokenize(text, format="text")
    return segmented.replace("_", " ")


_WORD_RE = re.compile(r"\w")


def tokenize_for_bm25(text: str) -> list[str]:
    """Segment + lowercase + bỏ token chỉ gồm dấu câu/ký hiệu markdown (|, **, #, -).

    Lowercase để "Nghỉ phép" (đầu câu) khớp "nghỉ phép"; bỏ ký hiệu để các token như
    "|" của bảng không chiếm IDF và làm nhiễu điểm.
    """
    return [t for t in segment_vietnamese(text).lower().split() if _WORD_RE.search(t)]



class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [tokenize_for_bm25(c["text"]) for c in chunks]
        # BM25Okapi chia cho số doc → corpus rỗng sẽ lỗi
        self.bm25 = BM25Okapi(self.corpus_tokens) if chunks else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                         metadata=self.documents[i].get("metadata", {}), method="bm25")
            for i in top_indices if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # recreate_collection() đã deprecated → xoá rồi tạo lại để index luôn sạch
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, batch_size=16, normalize_embeddings=True,
                                             show_progress_bar=len(texts) > 50)
        points = [
            PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(collection, points[start:start + 256])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip() or not self.client.collection_exists(collection):
            return []
        query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        # qdrant-client >= 1.9: query_points() thay cho search()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)
        results = []
        for pt in response.points:
            payload = dict(pt.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text=text, score=float(pt.score), metadata=payload, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1), rank bắt đầu từ 0.

    Chỉ dùng thứ hạng nên không cần chuẩn hoá điểm BM25 (không giới hạn) và cosine (≤1)
    về cùng thang. Doc nhận diện bằng text — cùng một chunk từ BM25 và Dense có cùng text.
    """
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)[:top_k]
    return [
        SearchResult(text=e["result"].text, score=e["score"], metadata=e["result"].metadata, method="hybrid")
        for e in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
