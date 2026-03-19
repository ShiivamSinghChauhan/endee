"""
RAG Retrieval + Generation Chain (LangChain + Groq)
-----------------------------------------------------
Builds a LangChain LCEL pipeline:
    User query
       │
       ▼
    HuggingFace Embedder ──► Endee ANN Search (top-k chunks)
                                    │
                                    ▼
                            Prompt Template
                                    │
                                    ▼
                            Groq LLM (llama3-8b)
                                    │
                                    ▼
                            Grounded Answer + Sources

Usage:
    from rag.retriever import RAGChain
    rag = RAGChain()
    answer = rag.ask("What is HNSW?")
    print(answer["result"])
    print(answer["source_documents"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag.endee_vectorstore import EndeeVectorStore


# ── Config ───────────────────────────────────────────────────────────────────
INDEX_NAME      = os.getenv("ENDEE_INDEX_NAME",  "rag-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL      = os.getenv("GROQ_MODEL",        "llama3-8b-8192")
TOP_K           = int(os.getenv("RETRIEVAL_TOP_K", "5"))


# ── System prompt ─────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a precise, helpful assistant that answers questions based strictly on the provided context.

Rules:
- Answer ONLY using the information in the context.
- If the answer isn't in the context, say: "I couldn't find that in the uploaded documents."
- Always cite the source document name (e.g., "According to paper.pdf, ...").
- Be concise and factual. Never hallucinate.
"""),
    ("human", """\
Context:
{context}

---

Question: {question}
"""),
])


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class RAGResponse:
    question:   str
    answer:     str
    sources:    list[dict] = field(default_factory=list)
    model_used: str = GROQ_MODEL


# ── RAG Chain ─────────────────────────────────────────────────────────────────
class RAGChain:
    """
    Full LangChain LCEL-based RAG chain:
        Endee (retrieval) + Groq LLaMA3 (generation)
    """

    def __init__(
        self,
        endee_host: str | None = None,
        endee_api_key: str | None = None,
        groq_api_key: str | None = None,
        index_name: str = INDEX_NAME,
        top_k: int = TOP_K,
        groq_model: str = GROQ_MODEL,
    ):
        self.index_name = index_name
        self.top_k      = top_k
        self.groq_model = groq_model

        # 1. HuggingFace Embeddings (free, local, no API key)
        print(f"[RAGChain] Loading embedding model: {EMBEDDING_MODEL} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Endee VectorStore
        self.vectorstore = EndeeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            endee_host=endee_host,
            endee_api_key=endee_api_key,
        )

        # 3. Groq LLM
        self.llm = ChatGroq(
            api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
            model=groq_model,
            temperature=0.2,
            max_tokens=1024,
        )

        # 4. Build LCEL chain
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        self._chain = self._build_chain()

    # ── Chain builder ─────────────────────────────────────────────────────

    def _build_chain(self):
        """
        LCEL pipeline:
            question ──► retriever ──► format context ──► prompt ──► LLM ──► str
        """
        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )

        chain = (
            {
                "context":  self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ── Public API ────────────────────────────────────────────────────────

    def ask(self, question: str, source_filter: str | None = None) -> RAGResponse:
        """
        Full RAG: retrieve from Endee → generate with Groq.

        Args:
            question      : User's natural language question
            source_filter : Optionally restrict to one document filename

        Returns:
            RAGResponse with .answer and .sources
        """
        # Apply optional source filter
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.top_k,
                **({"filter": {"source": {"$eq": source_filter}}} if source_filter else {}),
            },
        )

        # Retrieve source documents for citation display
        source_docs = retriever.invoke(question)

        # Run full chain
        answer = self._chain.invoke(question)

        sources = [
            {
                "source":      doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "?"),
                "text":        doc.page_content[:300],
            }
            for doc in source_docs
        ]

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            model_used=self.groq_model,
        )

    def stream(self, question: str, source_filter: str | None = None):
        """
        Generator that streams answer tokens for real-time UI display.
        Yields str tokens, then a final dict {"__sources__": [...]}
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.top_k,
                **({"filter": {"source": {"$eq": source_filter}}} if source_filter else {}),
            },
        )

        source_docs = retriever.invoke(question)

        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )

        context = format_docs(source_docs)

        messages = RAG_PROMPT.format_messages(context=context, question=question)

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

        # Sentinel with sources
        yield {
            "__sources__": [
                {
                    "source":      d.metadata.get("source", "unknown"),
                    "chunk_index": d.metadata.get("chunk_index", "?"),
                    "text":        d.page_content[:300],
                }
                for d in source_docs
            ]
        }
