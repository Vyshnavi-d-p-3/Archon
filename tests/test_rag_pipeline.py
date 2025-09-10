"""Tests for the RAG pipeline components."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import ToolCall
from tools.rag_pipeline import (
    BM25Retriever,
    DocumentChunk,
    HybridRetriever,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveChunker,
    TFIDFEmbedder,
    register_rag_tools,
)
from tools.registry import ToolRegistry


SAMPLE_DOC = """
Machine learning is a subset of artificial intelligence that focuses on
building systems that learn from data. Deep learning is a subset of machine
learning that uses neural networks with many layers.

Supervised learning uses labeled data to train models. Common algorithms
include linear regression, decision trees, and support vector machines.
Unsupervised learning finds patterns in unlabeled data using clustering
and dimensionality reduction techniques.

Reinforcement learning trains agents to make decisions by rewarding
desired behaviors. It has been successfully applied in game playing,
robotics, and autonomous vehicles. AlphaGo, developed by DeepMind,
famously defeated the world champion in the game of Go.

Transfer learning allows models trained on one task to be adapted for
another related task, significantly reducing the amount of training data
needed. Pre-trained models like BERT, GPT, and ResNet have revolutionized
natural language processing and computer vision.
"""

SAMPLE_DOC_2 = """
Python is a high-level programming language known for its readability.
It supports multiple programming paradigms including procedural,
object-oriented, and functional programming. The language was created
by Guido van Rossum and first released in 1991.

Python's ecosystem includes powerful libraries for data science such as
NumPy for numerical computing, Pandas for data manipulation, and
scikit-learn for machine learning. TensorFlow and PyTorch are popular
deep learning frameworks available in Python.
"""


# ── Chunker Tests ────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_DOC, source="test.txt")
        assert len(chunks) > 1
        assert all(c.source == "test.txt" for c in chunks)
        assert all(c.chunk_id for c in chunks)

    def test_chunk_ids_unique(self):
        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk(SAMPLE_DOC, source="test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_sequential(self):
        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk(SAMPLE_DOC, source="test")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_preserved(self):
        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.chunk(SAMPLE_DOC, metadata={"author": "test"})
        assert all(c.metadata.get("author") == "test" for c in chunks)

    def test_empty_content_filtered(self):
        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk("", source="empty")
        assert len(chunks) == 0

    def test_small_document_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk("Hello world.", source="tiny")
        assert len(chunks) == 1


# ── Embedder Tests ───────────────────────────────────────────────────

class TestTFIDFEmbedder:
    def test_fit_and_embed(self):
        embedder = TFIDFEmbedder(vocab_size=100)
        embedder.fit(["hello world", "machine learning is great"])
        vec = embedder.embed("hello world")
        assert vec.shape[0] > 0
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 0.01  # L2 normalized

    def test_similar_docs_closer(self):
        embedder = TFIDFEmbedder(vocab_size=200)
        docs = [
            "machine learning algorithms neural network training",
            "deep learning neural networks backpropagation",
            "cooking recipes pasta sauce italian food",
            "supervised learning classification regression models",
            "baking bread flour yeast oven temperature",
            "reinforcement learning reward policy gradient",
        ]
        embedder.fit(docs)

        v_ml = embedder.embed("machine learning models training")
        v_dl = embedder.embed("deep neural network learning")
        v_cook = embedder.embed("how to cook pasta baking")

        # ML and DL should be more similar than ML and cooking
        sim_ml_dl = float(np.dot(v_ml, v_dl))
        sim_ml_cook = float(np.dot(v_ml, v_cook))
        assert sim_ml_dl > sim_ml_cook

    def test_embed_batch(self):
        embedder = TFIDFEmbedder(vocab_size=50)
        embedder.fit(["hello", "world"])
        batch = embedder.embed_batch(["hello", "world", "test"])
        assert batch.shape[0] == 3

    def test_auto_fit(self):
        embedder = TFIDFEmbedder(vocab_size=50)
        # Should auto-fit without explicit fit()
        vec = embedder.embed("hello world test")
        assert vec.shape[0] > 0


# ── Vector Store Tests ───────────────────────────────────────────────

class TestInMemoryVectorStore:
    def _make_chunks(self, n: int = 5) -> list[DocumentChunk]:
        embedder = TFIDFEmbedder(vocab_size=50)
        texts = [f"Document about topic {i} with content" for i in range(n)]
        embedder.fit(texts)
        return [
            DocumentChunk(
                chunk_id=f"chunk_{i}",
                content=texts[i],
                embedding=embedder.embed(texts[i]),
            )
            for i in range(n)
        ]

    def test_add_and_size(self):
        store = InMemoryVectorStore()
        chunks = self._make_chunks(5)
        added = store.add(chunks)
        assert added == 5
        assert store.size == 5

    def test_search_returns_results(self):
        store = InMemoryVectorStore()
        chunks = self._make_chunks(5)
        store.add(chunks)
        query = chunks[0].embedding
        results = store.search(query, top_k=3)
        assert len(results) <= 3
        assert results[0].chunk.chunk_id == "chunk_0"  # Most similar to itself

    def test_get_by_id(self):
        store = InMemoryVectorStore()
        chunks = self._make_chunks(3)
        store.add(chunks)
        found = store.get_by_id("chunk_1")
        assert found is not None
        assert found.chunk_id == "chunk_1"

    def test_get_nonexistent(self):
        store = InMemoryVectorStore()
        assert store.get_by_id("missing") is None

    def test_clear(self):
        store = InMemoryVectorStore()
        store.add(self._make_chunks(3))
        store.clear()
        assert store.size == 0

    def test_search_empty_store(self):
        import numpy as np
        store = InMemoryVectorStore()
        results = store.search(np.zeros(10), top_k=5)
        assert results == []


# ── BM25 Tests ───────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_basic_search(self):
        chunks = [
            DocumentChunk(chunk_id="1", content="Python programming language"),
            DocumentChunk(chunk_id="2", content="Java programming language"),
            DocumentChunk(chunk_id="3", content="Cooking recipes for pasta"),
        ]
        bm25 = BM25Retriever()
        bm25.index(chunks)
        results = bm25.search("Python programming", top_k=2)
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "1"

    def test_no_match_returns_empty(self):
        chunks = [DocumentChunk(chunk_id="1", content="hello world")]
        bm25 = BM25Retriever()
        bm25.index(chunks)
        results = bm25.search("xyz123nonexistent", top_k=5)
        assert len(results) == 0

    def test_empty_index(self):
        bm25 = BM25Retriever()
        results = bm25.search("anything", top_k=5)
        assert results == []


# ── RAG Pipeline Integration ────────────────────────────────────────

class TestRAGPipeline:
    def test_ingest_and_search(self):
        pipeline = RAGPipeline(chunk_size=50, vocab_size=200)
        count = pipeline.ingest(SAMPLE_DOC, source="ml_guide.txt")
        assert count > 0
        assert pipeline.document_count > 0

        results = pipeline.retrieve("neural networks deep learning", top_k=3)
        assert len(results) > 0
        assert results[0].retrieval_method == "hybrid"

    def test_multi_document_ingest(self):
        pipeline = RAGPipeline(chunk_size=50, vocab_size=200)
        pipeline.ingest(SAMPLE_DOC, source="ml.txt")
        pipeline.ingest(SAMPLE_DOC_2, source="python.txt")
        assert pipeline.document_count > 2  # Multiple chunks each

        ml_results = pipeline.retrieve("supervised learning algorithms")
        py_results = pipeline.retrieve("Guido van Rossum Python")
        assert len(ml_results) > 0
        assert len(py_results) > 0

    def test_retrieve_as_context(self):
        pipeline = RAGPipeline(chunk_size=50, vocab_size=200)
        pipeline.ingest(SAMPLE_DOC, source="test.txt")
        context = pipeline.retrieve_as_context("reinforcement learning", max_tokens=500)
        assert len(context) > 0
        assert "Source:" in context

    def test_empty_pipeline_returns_nothing(self):
        pipeline = RAGPipeline()
        results = pipeline.retrieve("anything")
        assert results == []


# ── Tool Registration ────────────────────────────────────────────────

class TestRAGToolRegistration:
    def test_register_rag_tools(self):
        registry = ToolRegistry()
        register_rag_tools(registry)
        assert "rag_ingest" in registry
        assert "rag_search" in registry
        assert "rag_context" in registry

    def test_ingest_tool_execution(self):
        from tools.rag_pipeline import _pipeline
        import tools.rag_pipeline as rp
        rp._pipeline = None  # Reset singleton

        registry = ToolRegistry()
        register_rag_tools(registry)

        call = ToolCall(
            tool_name="rag_ingest",
            arguments={"text": SAMPLE_DOC, "source": "test.txt"},
        )
        result = registry.execute_tool_call(call)
        assert result.success
        assert result.output["chunks_added"] > 0

    def test_search_tool_execution(self):
        import tools.rag_pipeline as rp
        rp._pipeline = None  # Reset singleton

        registry = ToolRegistry()
        register_rag_tools(registry)

        # Ingest first
        registry.execute_tool_call(ToolCall(
            tool_name="rag_ingest",
            arguments={"text": SAMPLE_DOC, "source": "test"},
        ))

        # Then search
        result = registry.execute_tool_call(ToolCall(
            tool_name="rag_search",
            arguments={"query": "neural networks", "top_k": 3},
        ))
        assert result.success
        assert len(result.output) > 0

    def test_default_registry_includes_rag(self):
        from tools.implementations import build_default_registry
        registry = build_default_registry(use_mock=True, include_rag=True)
        assert "rag_ingest" in registry
        assert "rag_search" in registry
        assert "rag_context" in registry
        assert len(registry) == 9  # 6 original + 3 RAG


# Need numpy for vector tests
import numpy as np


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
