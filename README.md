# Benna AI
### Construction Document Intelligence for the GCC

Benna AI is an open-source, locally-deployable AI platform for construction professionals working on GCC projects. It lets you query contracts, specifications, and RFIs in **Arabic and English**, automatically detect conflicts between documents, verify delay claims against real-world weather and holiday data, and scan contracts for risk exposure — all from a single Streamlit interface.

No cloud subscriptions required. Run fully offline with Ollama, or connect to Claude via API.

---

## Features

### 🗂️ Document Intelligence & Q&A
- **Bilingual** — Arabic and English in the same project, same query
- **Hybrid retrieval** — semantic (ChromaDB + multilingual-e5-large) + keyword (BM25) fused with Reciprocal Rank Fusion (RRF)
- **Structure-aware chunking** — respects FIDIC/NEC clause hierarchies, ALL CAPS headers, Arabic section markers (`المادة`, `البند`, `الفقرة`), and RFI boundaries
- **Query rewriting** — Claude automatically expands abbreviations (LD → liquidated damages, BOQ → bill of quantities) before retrieval
- **Source citations** — every answer links back to document, page, and clause reference
- **Embedding cache** — SHA-256 keyed diskcache skips re-embedding already-seen chunks

### ⚡ Conflict Detection
- Compare any two documents (or doc types) on any topic
- Structured verdict: **CONTRADICTION / ALIGNED / GAP / UNCLEAR**
- Side-by-side source chunk view with clause references

### 📊 Claims & Delay Analysis *(new)*
Automatic real-world verification when a query involves a delay event:
- **Historical weather data** via [Open-Meteo](https://open-meteo.com/) — max temperature, precipitation, wind speed for any past date and GCC city (no API key required)
- **Public holidays** via [Nager.Date](https://date.nager.at/) — checks whether the claimed date was a public holiday in the relevant country
- **Weekend detection** — identifies Friday/Saturday weekends in GCC context
- **Geocoding fallback** — resolves any city globally if not in the built-in GCC registry
- Results are injected as a verified context block into the LLM, and displayed as a visual **Verified Claims Data** card in the chat UI

### 🛡️ Contract Risk Scanner *(new)*
One-click automated risk scan of any uploaded contract against a curated GCC/FIDIC-aligned knowledge base:

| Risk Topic | Severity if Missing |
|---|---|
| Liquidated Damages Cap | 🔴 HIGH |
| Force Majeure | 🔴 HIGH |
| Dispute Resolution Mechanism | 🔴 HIGH |
| Payment Terms & Timelines | 🔴 HIGH |
| Extension of Time (EOT) Entitlement | 🔴 HIGH |
| Limitation of Liability | 🔴 HIGH |
| Defects Liability Period | 🟡 MEDIUM |
| Termination for Convenience | 🟡 MEDIUM |
| Variation / Change Order Mechanism | 🟡 MEDIUM |
| Governing Law & Jurisdiction | 🟡 MEDIUM |
| Insurance Requirements | 🟡 MEDIUM |
| Performance Security / Bond | 🟢 LOW |

- Color-coded risk cards (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW / ✅ OK)
- Summary metrics dashboard: High, Medium, Low/OK counts and missing clause count
- Expandable source clause excerpts per flag with page and clause references
- Knowledge base compiled from publicly available GCC/FIDIC legal commentary — no copyrighted text embedded

### ⚙️ Platform
- **Switchable LLM** — run fully local with Ollama (Qwen 2.5) or use Claude API
- **Mock LLM provider** — test the full UI and pipeline offline without any LLM
- **Per-project namespacing** — manage multiple projects with isolated indexes
- **Streamlit UI** — RTL-aware chat, streaming responses, filter by doc type / language
- **Excel support** — ingest `.xlsx`, `.xls`, `.xlsm` files alongside PDFs

---

## Demo

**Chat tab — ask questions:**
> *"What are the liquidated damages if the contractor is delayed?"*
>
> Benna AI → *Section 3.2.1 of Contract-TowerB.pdf, Page 14: "Liquidated damages shall be assessed at AED 5,000 per calendar day…"*

**Claims & Delay Analysis — automatically triggered:**
> *"Was the delay on October 12, 2025 in Dubai justified due to weather?"*
>
> Benna AI fetches live weather data → *Max Temp: 37.5°C, Precipitation: 0.0mm, Wind: 12.8km/h — Sunday (Weekend) — No public holiday* → LLM synthesizes the answer against the contract EOT clause.

**Conflict Detection tab — compare documents:**
> *Topic: "concrete grade requirements"* · Doc A: contract · Doc B: spec
>
> **CONTRADICTION DETECTED**
> *Document A specifies C30 concrete for structural columns (Clause 5.2.1). Document B requires C35 as minimum grade for all structural elements (Section 4.3). The contract is under-specified relative to the technical specification…*

**Risk Scan tab — scan a contract:**
> Scanning *Main Contract.pdf* → 
> 🔴 HIGH: *No Liquidated Damages cap found. Uncapped LDs expose the contractor to unlimited liability.*
> 🔴 HIGH: *Force Majeure clause missing — no protection for unforeseeable events.*
> 🟡 MEDIUM: *Defects Liability Period is 36 months — significantly above the standard 12-month norm.*

Arabic queries supported:
> *"ما هي شروط الدفع المحددة في العقد؟"*

---

## Architecture

```
PDF / Excel Upload
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Ingest Pipeline                                │
│  PyMuPDF → Language Detection →                 │
│  Structure-Aware Chunker →                      │
│  multilingual-e5-large Embeddings               │
│  (diskcache — skip re-embedding)                │
└───────────────┬─────────────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  ChromaDB            BM25 Index
 (vector store)     (keyword index)
       │                 │
       └────────┬────────┘
                │  Reciprocal Rank Fusion
                ▼
          Top Chunks
          ┌────────┬──────────────┬──────────────────┐
          ▼        ▼              ▼                  ▼
   Query        Conflict       Claims           Risk Scan
   Pipeline     Pipeline       Pipeline         Pipeline
   (Q&A)        (Doc A vs B)   (EOT/Weather)    (12-topic)
          │        │              │                  │
          ▼        ▼              ▼                  ▼
   LLM (Qwen / Claude / Mock)
          │
          ▼
   Answer + Citations + Verified Data + Risk Flags
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
| Excel parsing | openpyxl / xlrd |
| Language detection | langdetect |
| LLM (local) | Qwen 2.5 7B via Ollama |
| LLM (cloud) | Claude (claude-sonnet-4-6) via Anthropic API |
| LLM (testing) | Mock provider (built-in) |
| Orchestration | LangChain |
| Weather API | Open-Meteo (free, no key required) |
| Holidays API | Nager.Date (free, no key required) |

---

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed (for local LLM) **or** an Anthropic API key (for Claude)

### 1. Clone and run

```bash
git clone https://github.com/cse-ahsankhan/BennaAI.git
cd BennaAI
bash run.sh      # Linux / macOS
# or
.\run.bat        # Windows
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

# Offline testing (no LLM needed)
LLM_PROVIDER=mock
```

### 3. Pull the model (Ollama path)

```bash
ollama pull qwen2.5:7b
```

---

## Usage

### Chat Tab
1. **Create a project** in the sidebar (e.g. `tower-b-phase2`)
2. **Upload PDFs** (contracts, specs, RFIs) in the Documents tab
3. **Ask questions** in Arabic or English — claim queries automatically trigger weather/holiday verification

### Conflict Detection Tab
1. Select **Document A** and **Document B**
2. Enter the **topic** to compare (e.g. *"payment terms"*, *"concrete grade"*)
3. Click **Detect Conflicts**

### Risk Scan Tab *(new)*
1. Select any indexed contract document
2. Click **▶ Run Risk Scan**
3. Review color-coded risk flags across 12 standard GCC contract topics

---

## Project Structure

```
benna-ai/
├── run.bat / run.sh               # One-command launcher
├── push.sh                        # Git push helper
├── config.py                      # Central config (env vars)
├── ingest/
│   ├── loader.py                  # PDF + Excel extraction + language detection
│   ├── chunker.py                 # Structure-aware chunking (FIDIC, Arabic, RFI)
│   └── embedder.py                # multilingual-e5-large + diskcache
├── retrieval/
│   ├── vector_store.py            # ChromaDB wrapper + get_indexed_files
│   ├── bm25_index.py              # BM25 sparse index (Arabic-aware tokenizer)
│   └── hybrid.py                  # RRF fusion with metadata filtering
├── llm/
│   └── provider.py                # Ollama / Claude / Mock factory
├── pipeline/
│   ├── ingest_pipeline.py         # Ingest orchestration
│   ├── query_pipeline.py          # Query → retrieve → answer (streaming)
│   ├── conflict_pipeline.py       # Conflict detection: compare Doc A vs Doc B
│   ├── claims_helper.py           # Claims/delay: Open-Meteo + Nager.Date APIs
│   └── risk_pipeline.py           # Contract risk scanner: 12-topic knowledge base
├── app/
│   └── streamlit_app.py           # Streamlit UI (Chat + Documents + Conflict + Risk Scan)
└── data/                          # Gitignored — created at runtime
    ├── uploads/
    ├── chroma_db/
    ├── bm25_indexes/
    └── embed_cache/
```

---

## Example Queries

### Chat Tab

| Language | Query |
|---|---|
| English | `What are the payment terms in the contract?` |
| English | `Which concrete grade is required for the foundation slab?` |
| English | `Was the delay on Dec 15, 2024 in Riyadh justified due to weather?` |
| English | `What happens if the contractor is delayed by the employer?` |
| Arabic | `ما هي شروط الدفع المحددة في العقد؟` |
| Arabic | `ما هي درجة الخرسانة المطلوبة للبلاطة الأساسية؟` |

### Conflict Detection Tab

| Topic | Doc A | Doc B | Typical finding |
|---|---|---|---|
| `concrete grade` | contract | spec | Grade mismatch (C30 vs C35) |
| `payment terms` | contract | RFI | Conflicting milestone dates |
| `defects liability period` | contract | spec | Duration not addressed in spec |
| `insurance requirements` | contract | spec | Spec silent — gap identified |

### Risk Scan Tab

| Document | Typical findings |
|---|---|
| Any GCC contract | Missing LD cap, narrow EOT entitlement, on-demand performance bond |
| Employer-drafted contract | No limitation of liability clause, extended DLP (36 months), no interest on late payments |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `claude`, or `mock` |
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

## Roadmap & Future Enhancements

### 🔜 In Progress
- **Semantic Diffing (Multi-Version Reconciliation)** — Upload two versions of the same specification and get a semantic diff: not just what words changed, but what the change *means* for procurement, obligations, or risk (e.g. "Concrete grade upgraded from C30 to C35 in Clause 4.3 — affects material cost and BOQ pricing").
- **Site Report Entity Extraction + Knowledge Graph** — Extract structured entities (Material, Location, Actor, Date, Event) from daily site reports and link them to contract clauses via a lightweight graph (networkx). Answering "what caused the delays in Zone B?" traverses the graph to surface related contract obligations.

### 📋 Planned
- **Automated benchmarks** — precision@k, MRR, latency, and token-cost comparison across retrieval configurations
- **Graph-based retrieval expansion** — Expand candidate chunks using graph neighbor links (graphify integration)
- **Neo4j / RedisGraph** — Production-grade graph store option for enterprise deployments
- **Multi-modal support** — Extract text from scanned PDFs and images using OCR (Tesseract / EasyOCR)
- **BOQ / Schedule analysis** — Parse and query Bill of Quantities Excel files with financial context-awareness
- **Clause template library** — Pre-ingested reference clauses from publicly available standard GCC addenda for use as comparison baselines in conflict detection
- **Progressive Web App (PWA)** — Mobile-friendly field access for site engineers
- **Audit trail & annotation** — Allow users to annotate flagged clauses and export a risk report as PDF
- **Multi-project dashboard** — Cross-project risk summary and document index statistics
- **Fine-tuned embeddings** — Domain-adapted embedding model trained on GCC construction clause pairs for improved retrieval precision

---

## License

MIT
