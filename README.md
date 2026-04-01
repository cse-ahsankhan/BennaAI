# Benna AI 🏗️
### Construction Document Intelligence for the GCC

Benna AI lets construction professionals query project documents — contracts, technical specs, and RFIs — using natural language in **Arabic and English**. Upload a PDF, ask a question, get an answer with the exact source clause cited.

Built with hybrid retrieval (semantic search + BM25) fused via Reciprocal Rank Fusion, and a dedicated **Conflict Detection** engine that compares two documents side-by-side to surface contradictions, gaps, and alignments.

---

## Features

- **Bilingual** — Arabic and English in the same project, same query
- **Hybrid retrieval** — semantic (ChromaDB + multilingual-e5-large) + keyword (BM25) fused with RRF
- **Structure-aware chunking** — respects FIDIC/NEC clause hierarchies, ALL CAPS headers, Arabic section markers (`المادة`, `البند`, `الفقرة`), and RFI boundaries
- **Conflict Detection** — compare any two documents (or doc types) on any topic; get a structured verdict: CONTRADICTION / ALIGNED / GAP / UNCLEAR
- **Query rewriting** — Claude automatically expands abbreviations (LD → liquidated damages, BOQ → bill of quantities) before retrieval
- **Embedding cache** — SHA-256 keyed diskcache skips re-embedding already-seen chunks
- **Source citations** — every answer links back to document, page, and clause reference
- **Switchable LLM** — run fully local with Ollama (Qwen 2.5) or use Claude API
- **Per-project namespacing** — manage multiple projects with isolated indexes
- **Streamlit UI** — RTL-aware chat, streaming responses, filter by doc type / language

---

## Demo

**Chat tab — ask questions:**
> *"What are the liquidated damages if the contractor is delayed?"*
>
> Benna AI → *Section 3.2.1 of Contract-TowerB.pdf, Page 14: "Liquidated damages shall be assessed at AED 5,000 per calendar day…"*

**Conflict Detection tab — compare documents:**
> *Topic: "concrete grade requirements"* · Doc A: contract · Doc B: spec
>
> ⚠️ **CONTRADICTION DETECTED**
> *Document A specifies C30 concrete for structural columns (Clause 5.2.1). Document B requires C35 as minimum grade for all structural elements (Section 4.3). The contract is under-specified relative to the technical specification…*

Arabic queries supported:
> *"ما هي شروط الدفع المحددة في العقد؟"*

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
│  (diskcache — skip re-embedding)        │
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
          Top Chunks
          ┌────────┴──────────────────────┐
          ▼                               ▼
   Query Pipeline                Conflict Pipeline
   (single answer)           (compare Doc A vs Doc B)
          │                               │
          ▼                               ▼
   LLM (Qwen / Claude)        LLM structured comparison
          │                               │
          ▼                               ▼
  Answer + Citations         Verdict + Analysis + Sources
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | `intfloat/multilingual-e5-large` via sentence-transformers |
| Embedding cache | diskcache (SHA-256 keyed) |
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

The run script handles everything: virtual environment, dependencies, and launching at `http://localhost:8501`.

> **First run** downloads `intfloat/multilingual-e5-large` (~1.1 GB). Allow a few minutes.

### 2. Configure

The run script creates `.env` from `.env.example` automatically. Edit it to switch providers:

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

### Chat
1. Open `http://localhost:8501`
2. **Create a project** in the sidebar (e.g. `tower-b-phase2`)
3. **Upload PDFs** — contracts, specs, RFIs
4. **Ask questions** in the Chat tab — Arabic or English
5. Expand **Sources** to see document, page, clause, and retrieval confidence

### Conflict Detection
1. Switch to the **⚡ Conflict Detection** tab
2. Select **Document A** and **Document B** — by type (`contract`, `spec`, `rfi`) or by specific file
3. Enter the **topic** to compare (e.g. *"payment terms"*, *"defects liability period"*)
4. Click **Detect Conflicts**
5. Benna returns a structured verdict with full analysis and source chunks from each document

---

## Project Structure

```
benna-ai/
├── run.bat / run.sh               # One-command launcher
├── config.py                      # Central config (env vars)
├── ingest/
│   ├── loader.py                  # PDF extraction + language detection
│   ├── chunker.py                 # Structure-aware chunking (FIDIC, Arabic, RFI)
│   └── embedder.py                # multilingual-e5-large + diskcache
├── retrieval/
│   ├── vector_store.py            # ChromaDB wrapper + get_indexed_files
│   ├── bm25_index.py              # BM25 sparse index (Arabic-aware tokenizer)
│   └── hybrid.py                  # RRF fusion with metadata filtering
├── llm/
│   └── provider.py                # Ollama / Claude factory
├── pipeline/
│   ├── ingest_pipeline.py         # Ingest orchestration
│   ├── query_pipeline.py          # Query → retrieve → answer (streaming)
│   └── conflict_pipeline.py       # Conflict detection: compare Doc A vs Doc B
├── app/
│   └── streamlit_app.py           # Streamlit UI (Chat + Conflict Detection tabs)
└── data/                          # Gitignored — created at runtime
    ├── uploads/
    ├── chroma_db/
    ├── bm25_indexes/
    └── embed_cache/
```

---

## Example Queries

### Chat tab

| Language | Query |
|---|---|
| English | `What are the payment terms in the contract?` |
| English | `Which concrete grade is required for the foundation slab?` |
| English | `What are the contractor's obligations under clause 8?` |
| Arabic | `ما هي شروط الدفع المحددة في العقد؟` |
| Arabic | `ما هي درجة الخرسانة المطلوبة للبلاطة الأساسية؟` |

### Conflict Detection tab

| Topic | Doc A | Doc B | Typical finding |
|---|---|---|---|
| `concrete grade` | contract | spec | Grade mismatch (C30 vs C35) |
| `payment terms` | contract | RFI | Conflicting milestone dates |
| `defects liability period` | contract | spec | Duration not addressed in spec |
| `insurance requirements` | contract | spec | Spec silent — gap identified |

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
