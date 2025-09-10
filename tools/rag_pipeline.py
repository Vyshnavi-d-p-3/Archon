"""
RAG Pipeline — Retrieval-Augmented Generation tools for Archon.

Provides document ingestion, chunking, embedding, vector storage,
and hybrid retrieval (dense + sparse) as first-class tools in
the agent's tool registry.

Architecture:
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Document    │────▶│   Chunker    │────▶│  Embedder    │
  │  Loader      │     │ (recursive)  │     │ (pluggable)  │
  └──────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                             ┌──────▼───────┐
                                             │ Vector Store  │
                                             │ (in-memory)   │
                                             └──────┬───────┘
                                                    │
                       ┌──────────────┐      ┌──────▼───────┐
                       │    BM25      │─────▶│   Hybrid     │
                       │  (sparse)    │      │  Retriever   │
                       └──────────────┘      │  (RRF fuse)  │
                                             └──────────────┘

Design notes:
  - Vector store is numpy-based (no external DB dependency).
  - Embeddings are pluggable: sentence-transformers, OpenAI, or TF-IDF fallback.
  - BM25 sparse retrieval for keyword matching.
  - Reciprocal Rank Fusion (RRF) merges dense + sparse results.
  - All components register as tools in the Archon registry.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field

from tools.registry import BaseTool, ToolRegistry

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DocumentChunk:
    """A chunk of text with metadata and optional embedding."""
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    source: str = ""
    chunk_index: int = 0

    @property
    def token_estimate(self) -> int:
        return len(self.content.split())


@dataclass
class RetrievalResult:
    """A single retrieval result with score and source info."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content[:500],
            "score": round(self.score, 4),
            "source": self.chunk.source,
            "method": self.retrieval_method,
            "metadata": self.chunk.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════
# 1. Document Chunker — recursive text splitting
# ═══════════════════════════════════════════════════════════════════════

class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators.
    Falls through to smaller separators when chunks exceed max_size.

    Separator hierarchy: paragraphs → sentences → words
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks."""
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunks = []
        for i, content in enumerate(merged):
            chunk_id = hashlib.sha256(
                f"{source}:{i}:{content[:50]}".encode()
            ).hexdigest()[:12]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=content.strip(),
                metadata=metadata or {},
                source=source,
                chunk_index=i,
            ))

        return [c for c in chunks if c.content]  # Drop empty

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        parts = text.split(sep)

        result = []
        for part in parts:
            if len(part.split()) <= self.chunk_size:
                result.append(part)
            else:
                # Too big — recurse with next separator
                result.extend(self._split_recursive(part, separators[1:]))
        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and add overlap between boundaries."""
        if not chunks:
            return []

        merged = []
        current = ""

        for chunk in chunks:
            candidate = (current + " " + chunk).strip() if current else chunk
            if len(candidate.split()) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        # Add overlap
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped = [merged[0]]
            for i in range(1, len(merged)):
                prev_words = merged[i - 1].split()
                overlap_text = " ".join(prev_words[-self.chunk_overlap:])
                overlapped.append(overlap_text + " " + merged[i])
            return overlapped

        return merged


# ═══════════════════════════════════════════════════════════════════════
# 2. Embedder — pluggable embedding backends
# ═══════════════════════════════════════════════════════════════════════

class TFIDFEmbedder:
    """
    TF-IDF based embedder — no external dependencies.
    Deterministic, fast, good enough for evaluation.
    Production: swap for SentenceTransformer or OpenAI embeddings.
    """

    def __init__(self, vocab_size: int = 5000):
        self._vocab_size = vocab_size
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and IDF from corpus."""
        word_counts: Counter[str] = Counter()
        doc_freq: Counter[str] = Counter()

        for doc in documents:
            words = self._tokenize(doc)
            word_counts.update(words)
            doc_freq.update(set(words))

        # Top-N vocabulary by frequency
        top_words = [w for w, _ in word_counts.most_common(self._vocab_size)]
        self._vocab = {w: i for i, w in enumerate(top_words)}

        # IDF: log(N / df)
        n_docs = len(documents)
        self._idf = np.zeros(len(self._vocab))
        for word, idx in self._vocab.items():
            df = doc_freq.get(word, 1)
            self._idf[idx] = math.log(n_docs / df + 1)

        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        if not self._fitted:
            # Auto-fit on single document (degenerate but safe)
            self.fit([text])

        words = self._tokenize(text)
        tf = np.zeros(len(self._vocab))

        word_count = Counter(words)
        for word, count in word_count.items():
            if word in self._vocab:
                tf[self._vocab[word]] = count

        # TF-IDF
        if len(words) > 0:
            tf = tf / len(words)
        tfidf = tf * self._idf  # type: ignore

        # L2 normalize
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        return tfidf

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])

    @property
    def dimension(self) -> int:
        return len(self._vocab) if self._vocab else self._vocab_size

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


# ═══════════════════════════════════════════════════════════════════════
# 3. Vector Store — numpy-based approximate nearest neighbor
# ═══════════════════════════════════════════════════════════════════════

class InMemoryVectorStore:
    """
    In-memory vector store using numpy for cosine similarity search.

    For production, swap with pgvector, Pinecone, or Weaviate.
    This implementation is sufficient for evaluation and development.
    """

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._id_index: dict[str, int] = {}

    def add(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks with pre-computed embeddings to the store."""
        new_chunks = [c for c in chunks if c.embedding is not None]
        if not new_chunks:
            return 0

        start_idx = len(self._chunks)
        for i, chunk in enumerate(new_chunks):
            self._id_index[chunk.chunk_id] = start_idx + i
            self._chunks.append(chunk)

        new_embeddings = np.array([c.embedding for c in new_chunks])

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.info("vectorstore.add", count=len(new_chunks), total=len(self._chunks))
        return len(new_chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Cosine similarity search."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Cosine similarity (embeddings are already L2 normalized)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores = self._embeddings @ query_norm

        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append(RetrievalResult(
                    chunk=self._chunks[idx],
                    score=score,
                    retrieval_method="dense",
                ))

        return results

    def get_by_id(self, chunk_id: str) -> DocumentChunk | None:
        idx = self._id_index.get(chunk_id)
        return self._chunks[idx] if idx is not None else None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings = None
        self._id_index.clear()

    @property
    def all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)


# ═══════════════════════════════════════════════════════════════════════
# 4. BM25 Sparse Retriever
# ═══════════════════════════════════════════════════════════════════════

class BM25Retriever:
    """
    BM25 sparse retrieval for keyword-based search.
    Complements dense retrieval for exact term matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._chunks: list[DocumentChunk] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._tokenized_docs: list[list[str]] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._tokenized_docs = [self._tokenize(c.content) for c in chunks]
        self._doc_lengths = [len(d) for d in self._tokenized_docs]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths else 1.0
        )

        self._doc_freqs = Counter()
        for doc in self._tokenized_docs:
            for word in set(doc):
                self._doc_freqs[word] += 1

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """BM25 scoring and ranking."""
        if not self._chunks:
            return []

        query_terms = self._tokenize(query)
        n_docs = len(self._chunks)
        scores = []

        for i, doc_tokens in enumerate(self._tokenized_docs):
            score = 0.0
            doc_len = self._doc_lengths[i]
            tf_counter = Counter(doc_tokens)

            for term in query_terms:
                if term not in tf_counter:
                    continue
                tf = tf_counter[term]
                df = self._doc_freqs.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * doc_len / self._avg_doc_length
                )
                score += idf * numerator / denominator

            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=self._chunks[i],
                score=scores[i],
                retrieval_method="sparse",
            )
            for i in top_indices
            if scores[i] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


# ═══════════════════════════════════════════════════════════════════════
# 5. Hybrid Retriever — RRF fusion of dense + sparse
# ═══════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) of dense and sparse retrieval.

    RRF score = Σ 1 / (k + rank_i) for each retrieval system.
    k=60 is the standard constant from the original paper.
    """

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        bm25: BM25Retriever,
        embedder: TFIDFEmbedder,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self._vector_store = vector_store
        self._bm25 = bm25
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval with RRF fusion."""
        # Dense retrieval
        query_embedding = self._embedder.embed(query)
        dense_results = self._vector_store.search(query_embedding, top_k=top_k * 2)

        # Sparse retrieval
        sparse_results = self._bm25.search(query, top_k=top_k * 2)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (
                self._dense_weight / (self._rrf_k + rank + 1)
            )
            chunk_map[cid] = result.chunk

        for rank, result in enumerate(sparse_results):
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (
                self._sparse_weight / (self._rrf_k + rank + 1)
            )
            chunk_map[cid] = result.chunk

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]  # type: ignore

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=rrf_scores[cid],
                retrieval_method="hybrid",
            )
            for cid in sorted_ids
        ]


# ═══════════════════════════════════════════════════════════════════════
# 6. RAG Pipeline — Orchestrates ingest + retrieval
# ═══════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Complete RAG pipeline: ingest documents, build index, retrieve context.
    Exposes methods that map to Archon tool calls.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        vocab_size: int = 3000,
    ):
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = TFIDFEmbedder(vocab_size=vocab_size)
        self.vector_store = InMemoryVectorStore()
        self.bm25 = BM25Retriever()
        self._retriever: HybridRetriever | None = None
        self._corpus_fitted = False

    def ingest(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest a document: chunk → embed → store."""
        chunks = self.chunker.chunk(text, source=source, metadata=metadata)

        if not self._corpus_fitted:
            self.embedder.fit([c.content for c in chunks])
            self._corpus_fitted = True

        for chunk in chunks:
            chunk.embedding = self.embedder.embed(chunk.content)

        count = self.vector_store.add(chunks)

        # Rebuild BM25 index
        self.bm25.index(self.vector_store.all_chunks)

        # Rebuild hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25=self.bm25,
            embedder=self.embedder,
        )

        logger.info("rag.ingest", source=source, chunks=count, total=self.vector_store.size)
        return count

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        if self._retriever is None:
            return []
        return self._retriever.retrieve(query, top_k=top_k)

    def retrieve_as_context(self, query: str, top_k: int = 5, max_tokens: int = 2000) -> str:
        """Retrieve and format as a context string for LLM prompts."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "(no relevant context found)"

        context_parts = []
        total_tokens = 0
        for r in results:
            chunk_tokens = r.chunk.token_estimate
            if total_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(
                f"[Source: {r.chunk.source} | Score: {r.score:.3f}]\n{r.chunk.content}"
            )
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    @property
    def document_count(self) -> int:
        return self.vector_store.size


# ═══════════════════════════════════════════════════════════════════════
# 7. Archon Tool Wrappers — Register RAG as agent tools
# ═══════════════════════════════════════════════════════════════════════

# Shared pipeline instance (singleton per registry)
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class IngestDocumentArgs(BaseModel):
    text: str = Field(description="Full text content of the document to ingest")
    source: str = Field(default="document", description="Source identifier (filename, URL, etc.)")


class IngestDocumentTool(BaseTool):
    """Ingest a document into the RAG knowledge base."""
    name = "rag_ingest"
    description = (
        "Ingest a document into the knowledge base for later retrieval. "
        "The document will be chunked, embedded, and indexed for semantic search. "
        "Use this to load reference material before answering questions about it."
    )
    args_schema = IngestDocumentArgs

    def execute(self, text: str, source: str = "document") -> dict[str, Any]:
        pipeline = _get_pipeline()
        chunks_added = pipeline.ingest(text, source=source)
        return {
            "status": "ingested",
            "chunks_added": chunks_added,
            "total_chunks": pipeline.document_count,
            "source": source,
        }


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Search query to find relevant information")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SemanticSearchTool(BaseTool):
    """Search the knowledge base for relevant information."""
    name = "rag_search"
    description = (
        "Search the ingested knowledge base using hybrid retrieval "
        "(semantic + keyword). Returns the most relevant text chunks "
        "ranked by relevance. Use after ingesting documents with rag_ingest."
    )
    args_schema = SemanticSearchArgs

    def execute(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        pipeline = _get_pipeline()
        results = pipeline.retrieve(query, top_k=top_k)
        return [r.to_dict() for r in results]


class RAGContextArgs(BaseModel):
    query: str = Field(description="Question to retrieve context for")
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="Maximum context tokens")


class RAGContextTool(BaseTool):
    """Retrieve formatted context for answering a question."""
    name = "rag_context"
    description = (
        "Retrieve relevant context from the knowledge base, formatted as a "
        "text block ready to insert into an LLM prompt. Returns the most "
        "relevant chunks concatenated with source attribution. "
        "Use this when you need to answer a question based on ingested documents."
    )
    args_schema = RAGContextArgs

    def execute(self, query: str, max_tokens: int = 2000) -> str:
        pipeline = _get_pipeline()
        return pipeline.retrieve_as_context(query, max_tokens=max_tokens)


def register_rag_tools(registry: ToolRegistry) -> None:
    """Register all RAG tools into an existing registry."""
    registry.register(IngestDocumentTool())
    registry.register(SemanticSearchTool())
    registry.register(RAGContextTool())
