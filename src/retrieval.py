"""Three retrieval methods over the Day 6 corpus (662 'B' chunks): dense, BM25 and hybrid (RRF).

* dense  : query embedding (BAAI/bge-base-en-v1.5, same query prefix as Day 6) -> exact cosine search in the
           Day 6 pgvector table. Real model, real store; any failure raises, there is no silent fallback.
* bm25   : Okapi BM25 over the same chunk texts (pure Python, no extra dependency).
* hybrid : Reciprocal Rank Fusion of the dense and BM25 rankings, score = sum 1 / (rrf_k + rank).

Every retriever has `search(query, top_k, filters) -> list[dict]` with the fields id, text, source_id, source,
page, metadata, score, rank (+ `components` for hybrid). `filters` is the user-supplied scope of Day 6 style,
e.g. {"document_id": "bizethics", "chapter": 6}; it is applied BEFORE ranking in all three methods.
No gold, notes or expected answer ever reaches this module.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge asymmetric query prefix (Day 6)
FILTER_KEYS = {"document_id", "chapter"}

# Small English stop-word list: BM25 already down-weights frequent terms, this only trims noise and index size.
STOPWORDS = frozenset("""a an and are as at be been but by can did do does for from had has have how i if in into is it its
of on or so than that the their them then there these they this to was we were what when where which who whom why will with
would you your""".split())


def load_env() -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    for key, value in os.environ.items():  # real environment wins over the file
        if key in ("DATABASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "CHUNK_STRATEGY"):
            values[key] = value
    return values


def load_chunks(path: Path | None = None) -> list[dict[str, Any]]:
    return json.loads((path or ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return [t for t in tokens if t not in STOPWORDS]


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = dict(filters or {})
    bad = set(filters) - FILTER_KEYS
    if bad:
        raise ValueError(f"unsupported filter keys {sorted(bad)}; allowed {sorted(FILTER_KEYS)}")
    return filters


def _in_scope(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    if "document_id" in filters and chunk["source_id"] != filters["document_id"]:
        return False
    return not ("chapter" in filters and chunk["metadata"].get("chapter") != filters["chapter"])


def _result(chunk: dict[str, Any], score: float, rank: int, **extra: Any) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": round(float(score), 6), "rank": rank, **extra}


# --------------------------------------------------------------------------------------------- BM25
class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.lengths = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.lengths) / len(chunks)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(self.doc_tokens):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        ranked = sorted(((s, i) for i, s in scores.items() if _in_scope(self.chunks[i], filters)),
                        key=lambda x: (-x[0], self.chunks[x[1]]["id"]))
        return [_result(self.chunks[i], s, r) for r, (s, i) in enumerate(ranked[:top_k], 1)]


# --------------------------------------------------------------------------------------------- dense
class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]], env: dict[str, str] | None = None) -> None:
        env = env or load_env()
        self.url = env.get("DATABASE_URL")
        if not self.url:
            raise RuntimeError("DATABASE_URL missing in .env")
        self.strategy = env.get("CHUNK_STRATEGY", "B")
        self.model_name = env.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.dimensions = int(env.get("EMBEDDING_DIMENSIONS", "768"))
        self.by_id = {c["id"]: c for c in chunks}
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # never fall back to anything else
            raise RuntimeError(f"cannot load embedding model {self.model_name!r}: {exc.__class__.__name__}: {exc}") from None
        if int(self._model.get_sentence_embedding_dimension()) != self.dimensions:
            raise RuntimeError("EMBEDDING_DIMENSIONS does not match the model output")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (self.strategy,))
            n = cur.fetchone()[0]
        if n != len(chunks):
            raise RuntimeError(f"pgvector holds {n} '{self.strategy}' chunks but chunks.json has {len(chunks)}: re-export or re-ingest")

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self.url, connect_timeout=5)
        except psycopg.Error as exc:
            raise RuntimeError(f"cannot connect to the Day 6 pgvector store ({exc.__class__.__name__}); is Docker up?") from None

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        where, params = ["strategy = %s"], [self.strategy]
        if "document_id" in filters:
            where.append("document_id = %s")
            params.append(filters["document_id"])
        if "chapter" in filters:
            where.append("chapter = %s")
            params.append(filters["chapter"])
        vec = "[" + ",".join(repr(float(x)) for x in self.embed_query(query)) + "]"
        sql = f"""SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM chunks
                  WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, [vec, *params, vec, top_k])
            rows = cur.fetchall()
        missing = [cid for cid, _ in rows if cid not in self.by_id]
        if missing:
            raise RuntimeError(f"pgvector returned chunk ids absent from chunks.json: {missing[:3]}")
        return [_result(self.by_id[cid], score, r) for r, (cid, score) in enumerate(rows, 1)]


# --------------------------------------------------------------------------------------------- hybrid
class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, rrf_k: int = 60, candidates: int = 50) -> None:
        self.dense, self.bm25, self.rrf_k, self.candidates = dense, bm25, rrf_k, candidates

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        lists = {"dense": self.dense.search(query, self.candidates, filters),
                 "bm25": self.bm25.search(query, self.candidates, filters)}
        fused: dict[str, float] = defaultdict(float)
        info: dict[str, dict[str, Any]] = {}
        for method, results in lists.items():
            for r in results:
                fused[r["id"]] += 1.0 / (self.rrf_k + r["rank"])
                info.setdefault(r["id"], {"chunk": r, "components": {}})["components"][method] = r["rank"]
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [_result(info[cid]["chunk"], score, rank, components=info[cid]["components"])
                for rank, (cid, score) in enumerate(ranked, 1)]


# --------------------------------------------------------------------------------------------- rerank
RERANKERS = {"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~88 MB cache
             "bge": "BAAI/bge-reranker-base"}  # ~1.1 GB cache


class RerankRetriever:
    """Hybrid pool (default 30 candidates) re-ordered by a local cross-encoder that reads (query, passage) together.

    The filter is applied inside the hybrid pool; the returned score is the cross-encoder score and `components` keeps the hybrid rank/score
    so before/after can be audited. No network call at query time; the model is downloaded once from the Hugging Face hub.
    """
    name = "rerank"

    def __init__(self, base: HybridRetriever, reranker: str = "minilm", pool: int = 30) -> None:
        self.base, self.pool = base, pool
        self.model_name = RERANKERS.get(reranker, reranker)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device="cpu")
        except Exception as exc:  # never fall back to another model silently
            raise RuntimeError(f"cannot load cross-encoder {self.model_name!r}: {exc.__class__.__name__}: {exc}") from None

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        candidates = self.base.search(query, max(self.pool, top_k), filters)
        if not candidates:
            return []
        scores = self._model.predict([(query, c["text"]) for c in candidates], show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["id"]))[:top_k]
        return [_result(candidates[i], float(scores[i]), rank, components={"hybrid_rank": candidates[i]["rank"], "hybrid_score": candidates[i]["score"]})
                for rank, i in enumerate(order, 1)]


def build_retrievers(chunks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    chunks = chunks or load_chunks()
    dense, bm25 = DenseRetriever(chunks), BM25Retriever(chunks)
    return {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever(dense, bm25)}
