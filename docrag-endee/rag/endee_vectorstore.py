"""
LangChain VectorStore Wrapper for Endee
---------------------------------------
Integrates Endee into the LangChain ecosystem as a first-class VectorStore.
This lets us use Endee with all LangChain chains, retrievers, and tooling.

Usage:
    from rag.endee_vectorstore import EndeeVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = EndeeVectorStore(index_name="rag-docs", embedding=embeddings)
    vectorstore.add_texts(["chunk 1 text", "chunk 2 text"], metadatas=[{...}, {...}])
    docs = vectorstore.similarity_search("What is RAG?", k=5)
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rag.endee_client import EndeeClient


BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))


def _chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic ID — re-ingesting same file stays idempotent."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class EndeeVectorStore(VectorStore):
    """
    LangChain-compatible VectorStore backed by Endee.

    Supports:
        - add_texts / add_documents
        - similarity_search
        - similarity_search_with_score
        - as_retriever() (inherited from VectorStore base)
    """

    def __init__(
        self,
        index_name: str,
        embedding: Embeddings,
        endee_host: str | None = None,
        endee_api_key: str | None = None,
        metric: str = "cosine",
        quantization: str = "FLOAT32",
    ):
        self.index_name  = index_name
        self.embedding   = embedding
        self.metric      = metric
        self.quantization = quantization
        self.client      = EndeeClient(host=endee_host, api_key=endee_api_key)
        self._dimension: int | None = None

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_dimension(self) -> int:
        """Probe embedding dimension by encoding a dummy string once."""
        if self._dimension is None:
            sample = self.embedding.embed_query("hello")
            self._dimension = len(sample)
        return self._dimension

    def _ensure_index(self) -> None:
        """Create the Endee index if it doesn't exist yet."""
        if self.client.index_exists(self.index_name):
            return
        dim = self._get_dimension()
        print(f"[EndeeVectorStore] Creating index '{self.index_name}' (dim={dim}) ...")
        self.client.create_index(
            name=self.index_name,
            dimension=dim,
            metric=self.metric,
            quantization=self.quantization,
        )
        print(f"[EndeeVectorStore] Index created ✓")

    # ── VectorStore interface ────────────────────────────────────────────

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Embed texts and upsert into Endee.
        Returns list of vector IDs.
        """
        self._ensure_index()

        texts_list = list(texts)
        metadatas  = metadatas or [{} for _ in texts_list]
        ids: list[str] = []

        # Process in batches
        for i in range(0, len(texts_list), BATCH_SIZE):
            batch_texts = texts_list[i : i + BATCH_SIZE]
            batch_meta  = metadatas[i : i + BATCH_SIZE]

            embeddings = self.embedding.embed_documents(batch_texts)

            vectors = []
            for j, (text, meta, emb) in enumerate(zip(batch_texts, batch_meta, embeddings)):
                vec_id = _chunk_id(
                    meta.get("source", "unknown"),
                    meta.get("chunk_index", i + j),
                )
                ids.append(vec_id)
                vectors.append({
                    "id":     vec_id,
                    "values": emb,
                    "metadata": {**meta, "text": text},
                })

            self.client.upsert(self.index_name, vectors)
            print(f"[EndeeVectorStore] Upserted {min(i + BATCH_SIZE, len(texts_list))}/{len(texts_list)} chunks")

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return top-k Documents most similar to the query."""
        results = self.similarity_search_with_score(query, k=k, filter=filter)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return top-k (Document, score) tuples."""
        query_vec = self.embedding.embed_query(query)
        raw       = self.client.search(
            index_name=self.index_name,
            query_vector=query_vec,
            top_k=k,
            include_metadata=True,
            filters=filter,
        )

        results = []
        for r in raw:
            meta  = r.get("metadata", {})
            text  = meta.pop("text", "")        # pull text out of metadata
            score = r.get("score", 0.0)
            doc   = Document(page_content=text, metadata=meta)
            results.append((doc, score))

        return results

    # ── LangChain factory method (required abstract) ─────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        index_name: str = "rag-docs",
        **kwargs: Any,
    ) -> "EndeeVectorStore":
        """Create an EndeeVectorStore and populate it from a list of texts."""
        store = cls(index_name=index_name, embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store
