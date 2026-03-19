# 📄 DocRAG — Chat With Your Document

> **RAG pipeline built on [Endee Vector Database](https://github.com/endee-io/endee)**  
> Endee.io Internship Project Submission — March 2026

<br>

![DocRAG Demo](https://img.shields.io/badge/Status-Working-22c55e?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-6366f1?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-f97316?style=flat-square)
![Endee](https://img.shields.io/badge/Endee-VectorDB-06d6a0?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-a78bfa?style=flat-square)

---

## What Is This?

**DocRAG** is a production-style **Retrieval-Augmented Generation (RAG)** system. Upload any PDF or TXT file, ask questions in natural language, and get accurate answers grounded in your document — with source citations and zero hallucination.

The system uses **Endee** as its vector database to store and search document embeddings at high speed using HNSW-based Approximate Nearest Neighbour (ANN) search.

---

## Demo

```
User  → "What is the leaving criteria?"

System → Embeds query using HuggingFace
       → Searches Endee index (cosine similarity, top-5 chunks)
       → Sends retrieved context to Groq LLaMA3
       → Returns grounded answer with source citations
```

> *"According to Sudhir Kumar Offer Letter.pdf, the company may terminate the contract without prior notice for serious misconduct or material breach..."*

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) | HNSW ANN search, cosine similarity |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Free, local, 384-dim vectors |
| **LLM** | Groq `llama-3.3-70b-versatile` | Fast, free answer generation |
| **Orchestration** | LangChain LCEL | Document loading, splitting, RAG chain |
| **UI** | Streamlit | Chat interface |

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                    INGESTION PIPELINE                        ║
║                                                              ║
║  PDF / TXT                                                   ║
║     │                                                        ║
║     ▼                                                        ║
║  LangChain PyPDFLoader / TextLoader                          ║
║     │                                                        ║
║     ▼                                                        ║
║  RecursiveCharacterTextSplitter                              ║
║  (chunk_size=800, overlap=150)                               ║
║     │                                                        ║
║     ▼                                                        ║
║  HuggingFace Embeddings (384-dim)                            ║
║     │                                                        ║
║     ▼                                                        ║
║  Endee Vector Index (HNSW, cosine)  ◄─── stored here        ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                     QUERY PIPELINE                           ║
║                                                              ║
║  User Question                                               ║
║     │                                                        ║
║     ▼                                                        ║
║  HuggingFace Embeddings (same model)                         ║
║     │                                                        ║
║     ▼                                                        ║
║  Endee ANN Search → Top-5 most relevant chunks              ║
║     │                                                        ║
║     ▼                                                        ║
║  LangChain LCEL Prompt                                       ║
║     │                                                        ║
║     ▼                                                        ║
║  Groq LLaMA3 (streaming)                                     ║
║     │                                                        ║
║     ▼                                                        ║
║  Grounded Answer + Source Citations                          ║
╚══════════════════════════════════════════════════════════════╝
```

---

## How Endee Is Used

DocRAG integrates Endee at three points via a custom LangChain `VectorStore` wrapper (`rag/endee_vectorstore.py`):

### 1. Creating the Index
```python
client.create_index(
    name="rag-docs",
    dimension=384,        # all-MiniLM-L6-v2 output size
    metric="cosine"
)
```

### 2. Inserting Vectors (after embedding chunks)
```python
client.upsert("rag-docs", [
    {
        "id":     "abc123",
        "vector": [0.021, -0.043, ...],   # 384-dim embedding
        "meta":   '{"text": "...", "source": "offer_letter.pdf"}',
    }
])
```

### 3. Searching at Query Time
```python
results = client.search(
    index_name="rag-docs",
    query_vector=query_embedding,
    k=5                              # top-5 nearest neighbours
)
# Returns MessagePack response decoded to list of matches
```

> **Note:** Endee's insert API expects `meta` as a JSON-encoded string, and the search response is binary MessagePack (decoded with the `msgpack` library).

---

## Project Structure

```
docrag-endee/
├── rag/
│   ├── endee_client.py         # Endee HTTP API wrapper (verified from source)
│   ├── endee_vectorstore.py    # LangChain VectorStore for Endee
│   ├── ingestion.py            # Load → split → embed → upsert pipeline
│   └── retriever.py            # LangChain LCEL RAG chain (Endee + Groq)
├── data/sample_docs/
│   └── intro_to_rag.txt        # Sample test document
├── app.py                      # Streamlit UI (upload → analyse → chat)
├── ingest.py                   # CLI ingestion tool
├── docker-compose.yml          # Endee + app services
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker
- **Groq API Key** — free at [console.groq.com](https://console.groq.com) (no credit card)

### Step 1 — Build & Start Endee

```bash
# Clone the Endee repo
git clone https://github.com/endee-io/endee.git
cd endee

# Build from source (Intel/AMD CPU)
docker build -t endee-oss:latest -f infra/Dockerfile --build-arg BUILD_ARCH=avx2 .

# Start Endee server
docker-compose up -d
# Endee dashboard → http://localhost:8080
```

### Step 2 — Set Up This Project

```bash
git clone https://github.com/<your-username>/docrag-endee.git
cd docrag-endee

pip install -r requirements.txt

cp .env.example .env
# Open .env → set GROQ_API_KEY=gsk_...
```

### Step 3 — Run

```bash
streamlit run app.py
# Open: http://localhost:8501
```

### Step 4 — Use It

1. Upload a PDF or TXT file
2. Click **✦ Analyse Document**
3. Ask anything about your document

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Free at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for generation |
| `ENDEE_HOST` | `http://localhost:8080` | Endee server URL |
| `ENDEE_INDEX_NAME` | `rag-docs` | Vector index name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |

---

## CLI Usage

```bash
# Ingest a single file from terminal
python ingest.py data/sample_docs/intro_to_rag.txt

# Ingest a whole folder
python ingest.py data/docs/
```

---

## Stop & Restart

```bash
# Stop everything
Ctrl+C                    # stop Streamlit
docker stop endee-oss     # stop Endee

# Start again
docker start endee-oss
streamlit run app.py
```

> Previously ingested vectors persist in Endee's Docker volume — no need to re-ingest.

---

## Submission Details

- **Role Applied:** SDE / AI / ML Intern
- **Endee Repo Starred:** ✅
- **Endee Repo Forked:** ✅
- **Submission Deadline:** 19 March 2026, 5 PM

---

## License

MIT
