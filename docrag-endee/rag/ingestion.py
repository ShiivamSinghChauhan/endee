"""
Ingestion Pipeline (LangChain-powered)
---------------------------------------
File → LangChain Document Loaders → RecursiveCharacterTextSplitter
     → HuggingFace Embeddings → EndeeVectorStore upsert
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from rag.endee_vectorstore import EndeeVectorStore


INDEX_NAME      = os.getenv("ENDEE_INDEX_NAME", "rag-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "150"))


class Ingestor:
    def __init__(
        self,
        endee_host: str | None = None,
        endee_api_key: str | None = None,
        index_name: str = INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.index_name = index_name

        print(f"[Ingestor] Loading embedding model: {embedding_model} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = EndeeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            endee_host=endee_host,
            endee_api_key=endee_api_key,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def _load_file(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
        return loader.load()

    def ingest_file(self, filepath: str | Path, display_name: str | None = None) -> int:
        """
        Ingest a single PDF or TXT file.
        display_name: the real filename to show in UI (overrides temp filename)
        Returns number of chunks indexed.
        """
        path = Path(filepath)
        real_name = display_name or path.name  # use real name if provided

        print(f"\n[Ingestor] Loading '{real_name}' ...")

        docs = self._load_file(path)

        chunks = self.splitter.split_documents(docs)
        print(f"[Ingestor] Split into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            # Fix words-on-separate-lines issue from PDF extraction
            chunk.page_content = " ".join(chunk.page_content.split())
            # Use real filename instead of temp path
            chunk.metadata["source"]      = real_name
            chunk.metadata["chunk_index"] = i

        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self.vectorstore.add_texts(texts, metadatas=metadatas)

        print(f"[Ingestor] ✅ '{real_name}' ingested — {len(chunks)} chunks in Endee.")
        return len(chunks)

    def ingest_directory(self, dirpath: str | Path, extensions=(".pdf", ".txt")) -> int:
        dirpath = Path(dirpath)
        files   = [f for f in sorted(dirpath.iterdir()) if f.suffix.lower() in extensions]

        if not files:
            print(f"[Ingestor] No {extensions} files found in '{dirpath}'.")
            return 0

        total = sum(self.ingest_file(f) for f in files)
        print(f"\n[Ingestor] ✅ Directory done. Total chunks ingested: {total}")
        return total