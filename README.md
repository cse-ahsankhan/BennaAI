# Benna AI 🏗️
### Construction Document Intelligence for the GCC

Benna AI lets construction professionals query project documents — contracts, technical specs, and RFIs — using natural language in **Arabic and English**. Upload a PDF, ask a question, get an answer with the exact source clause cited.

Built with hybrid retrieval (semantic search + BM25) fused via Reciprocal Rank Fusion, so it finds the right clause even when keyword and meaning diverge.

---

## Demo

> **"What are the liquidated damages if the contractor is delayed?"**
>
> *Benna AI → Section 3.2.1 of Contract-TowerB.pdf, Page 14: "Liquidated damages shall be assessed at AED 5,000 per calendar day..."*

Supports Arabic queries out of the box:
> **"ما هي شروط الدفع المحددة في العقد؟"**

---

## Features

- **Bilingual** — Arabic and English in the same project, same query
- **Hybrid retrieval** — semantic (ChromaDB + multilingual-e5-large) + keyword (BM25) fused with RRF
- **Structure-aware chunking** — respects clause hierarchies (`1.1`, `2.3.4`), section headers, and RFI boundaries
- **Source citations** — every answer links back to document, page, and clause reference
- **Switchable LLM** — run fully local with Ollama (Qwen 2.5 / Mistral) or use Claude API
- **Per-project namespacing** — manage multiple projects with isolated indexes
- **Streamlit UI** — RTL-aware chat interface, PDF upload, streaming responses

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│  Ingest Pipeline                        │
│  PyMuPDF → Language Detection →         │
│  Structure-Aware Chunker →              │
│  multilingual-e5-large Embeddings       │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ChromaDB            BM25 Index
 (vector store)      (keyword index)
        │                 │
        └────────┬────────┘
                 │  Reciprocal Rank Fusion
                 ▼
          Top 5 Chunks
                 │
                 ▼
         LLM (Qwen / Claude)
                 │
                 ▼
       Answer + Source Citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | `intfloat/multilingual-e5-large` via sentence-transformers |
| Vector store | ChromaDB (cosine similarity, persistent) |
| Keyword search | BM25Okapi (rank-bm25) |
| Retrieval fusion | Reciprocal Rank Fusion (RRF, k=60) |
| PDF parsing | PyMuPDF (fitz) |
| Language detection | langdetect |
| LLM (local) | Qwen 2.5 7B via Ollama |
| LLM (cloud) | Claude (claude-sonnet-4-6) via Anthropic API |
| Orchestration | LangChain |

---

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed (for local LLM) **or** an Anthropic API key (for Claude)

### 1. Clone and run

```bash
git clone https://github.com/your-username/benna-ai.git
cd benna-ai
.\run.bat        # Windows
# or
bash run.sh      # Linux / macOS
```

The run script handles everything: virtual environment, dependencies, and launching the app at `http://localhost:8501`.

> **First run** downloads `intfloat/multilingual-e5-large` (~1.1 GB). Allow a few minutes.

### 2. Configure

The run script creates `.env` automatically from `.env.example`. Edit it to switch providers:

```env
# Local (default)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b

# Cloud
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Pull the model (Ollama path)

```bash
ollama pull qwen2.5:7b
```

---

## Usage

1. Open `http://localhost:8501`
2. **Create a project** in the sidebar (e.g. `tower-b-phase2`)
3. **Upload a PDF** — contract, spec, or RFI
4. **Ask questions** in the chat — Arabic or English
5. Expand **Sources** below each answer to see document, page, and clause

---

## Project Structure

```
benna-ai/
├── run.bat / run.sh           # One-command launcher
├── config.py                  # Central config (env vars)
├── ingest/
│   ├── loader.py              # PDF extraction + language detection
│   ├── chunker.py             # Structure-aware chunking
│   └── embedder.py            # multilingual-e5-large embeddings
├── retrieval/
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── bm25_index.py          # BM25 sparse index
│   └── hybrid.py              # RRF fusion
├── llm/
│   └── provider.py            # Ollama / Claude factory
├── pipeline/
│   ├── ingest_pipeline.py     # Ingest orchestration
│   └── query_pipeline.py      # Query orchestration
├── app/
│   └── streamlit_app.py       # Streamlit UI
└── data/                      # Gitignored — created at runtime
    ├── uploads/
    ├── chroma_db/
    └── bm25_indexes/
```

---

## Example Queries

| Language | Query |
|---|---|
| English | `What are the payment terms in the contract?` |
| English | `Which concrete grade is required for the foundation slab?` |
| English | `What are the contractor's obligations under clause 8?` |
| Arabic | `ما هي شروط الدفع المحددة في العقد؟` |
| Arabic | `ما هي درجة الخرسانة المطلوبة للبلاطة الأساسية؟` |
| Arabic | `ملخص طلبات الاستفسار المتعلقة بالتنسيق الميكانيكي` |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `claude` |
| `ANTHROPIC_API_KEY` | — | Required for Claude |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any Ollama model |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence-transformer model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector store path |
| `BM25_INDEX_DIR` | `./data/bm25_indexes` | BM25 index path |
| `CHUNK_SIZE` | `512` | Chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (tokens) |

---

## Why Qwen 2.5 for Arabic?

Most open-source models have weak Arabic support. Qwen 2.5 (Alibaba) is trained on a significantly larger Arabic corpus than Mistral or Llama, making it the strongest locally-runnable option for GCC construction documents that mix Arabic and English.

---

## License

MIT
