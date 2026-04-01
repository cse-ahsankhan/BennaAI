"""
Benna AI — Construction Document Intelligence
Streamlit UI
"""
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from retrieval import vector_store  # noqa: E402
from pipeline.ingest_pipeline import ingest_document  # noqa: E402
from pipeline.query_pipeline import query_stream  # noqa: E402
from pipeline.conflict_pipeline import detect_conflicts  # noqa: E402
from llm.provider import clear_cache  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Benna AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        border-radius: 6px;
        padding: 2px 9px;
        font-size: 0.78em;
        font-weight: 600;
        margin: 0 2px;
    }
    .badge-contract { background:#DBEAFE; color:#1D4ED8; }
    .badge-spec     { background:#FEF3C7; color:#B45309; }
    .badge-rfi      { background:#D1FAE5; color:#065F46; }
    .badge-general  { background:#F3F4F6; color:#374151; }
    .badge-clause   { background:#EDE9FE; color:#5B21B6; }
    .section-header { font-style: italic; color: #6B7280; font-size: 0.88em; }
    .rewritten-query { font-style: italic; color: #9CA3AF; font-size: 0.84em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOC_TYPE_BADGE = {
    "contract": '<span class="badge badge-contract">Contract</span>',
    "spec":     '<span class="badge badge-spec">Spec</span>',
    "rfi":      '<span class="badge badge-rfi">RFI</span>',
    "general":  '<span class="badge badge-general">General</span>',
}


def _doc_badge(doc_type: str) -> str:
    return _DOC_TYPE_BADGE.get(doc_type, _DOC_TYPE_BADGE["general"])


def _clause_badge(clause_ref: str) -> str:
    if not clause_ref:
        return ""
    return f'<span class="badge badge-clause">📎 {clause_ref}</span>'


def _build_filters() -> Optional[Dict[str, str]]:
    """Convert sidebar filter widgets into a filters dict for the query pipeline."""
    filters: Dict[str, str] = {}

    doc_types = st.session_state.get("filter_doc_types", [])
    if doc_types and "All" not in doc_types:
        # Only single doc_type filter supported; take first selection
        filters["doc_type"] = doc_types[0].lower()

    lang = st.session_state.get("filter_language", "All")
    if lang == "Arabic only":
        filters["language"] = "ar"
    elif lang == "English only":
        filters["language"] = "en"

    return filters if filters else None


def _render_sources(sources: List[Dict]) -> None:
    """Render the enhanced sources expander content."""
    for i, src in enumerate(sources, 1):
        doc_type = src.get("doc_type", "general")
        clause_ref = src.get("clause_ref", "")
        section_header = src.get("section_header", "")
        rrf_score = src.get("rrf_score", 0.0)
        origins = " + ".join(sorted(set(src.get("retrieval_sources", []))))

        header_html = (
            f'{_doc_badge(doc_type)} {_clause_badge(clause_ref)} '
            f'<strong>{src["file"]}</strong> · Page {src["page"]} '
            f'<span style="color:#9CA3AF;font-size:0.82em">via {origins}</span>'
        )
        st.markdown(f"**[{i}]** " + header_html, unsafe_allow_html=True)

        if section_header:
            st.markdown(
                f'<div class="section-header">§ {section_header}</div>',
                unsafe_allow_html=True,
            )

        if src.get("text_snippet"):
            st.code(src["text_snippet"] + " …", language=None)

        confidence = min(rrf_score * 10, 1.0)
        st.progress(confidence, text=f"Relevance: {rrf_score:.4f}")
        if i < len(sources):
            st.divider()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = config.LLM_PROVIDER
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "embedding_warned" not in st.session_state:
    st.session_state.embedding_warned = False
if "filter_doc_types" not in st.session_state:
    st.session_state.filter_doc_types = ["All"]
if "filter_language" not in st.session_state:
    st.session_state.filter_language = "All"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏗️ Benna AI")
    st.caption("Construction Document Intelligence · GCC")
    st.divider()

    # --- LLM provider ---
    st.subheader("LLM Provider")
    provider_choice = st.radio(
        "Select LLM",
        options=["ollama", "claude"],
        index=0 if st.session_state.llm_provider == "ollama" else 1,
        horizontal=True,
        key="provider_radio",
    )
    if provider_choice != st.session_state.llm_provider:
        st.session_state.llm_provider = provider_choice
        clear_cache()
        st.success(f"Switched to {provider_choice}")

    if provider_choice == "claude" and not config.ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY is not set in your .env file.")

    st.divider()

    # --- Project selector ---
    st.subheader("Project")
    existing_projects = vector_store.list_projects()

    new_project_name = st.text_input(
        "New project name",
        placeholder="e.g. tower-b-phase2",
        help="Alphanumeric and hyphens only",
    )
    if st.button("Create project") and new_project_name.strip():
        sanitized = new_project_name.strip().lower().replace(" ", "-")
        st.session_state.project_id = sanitized
        st.session_state.messages = []
        st.success(f"Project '{sanitized}' created.")
        existing_projects = vector_store.list_projects()

    if existing_projects:
        selected = st.selectbox(
            "Or select existing project",
            options=["— select —"] + existing_projects,
        )
        if selected != "— select —" and selected != st.session_state.project_id:
            st.session_state.project_id = selected
            st.session_state.messages = []
    else:
        st.info("No projects yet — create one above.")

    st.divider()

    # --- Document upload ---
    st.subheader("Upload Document")
    if not st.session_state.project_id:
        st.warning("Select or create a project first.")
    else:
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Contracts, technical specs, RFIs — Arabic or English",
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = Path(tmp.name)

            progress_placeholder = st.empty()
            messages_log: List[str] = []

            def _progress(msg: str) -> None:
                messages_log.append(msg)
                progress_placeholder.info("\n\n".join(messages_log[-3:]))

            with st.spinner("Ingesting document …"):
                try:
                    if not st.session_state.embedding_warned:
                        st.toast(
                            "First run: downloading embedding model (~1.1 GB). "
                            "This may take a few minutes.",
                            icon="⚠️",
                        )
                        st.session_state.embedding_warned = True
                    summary = ingest_document(
                        tmp_path,
                        st.session_state.project_id,
                        progress_callback=_progress,
                    )
                    progress_placeholder.success(
                        f"Ingested **{summary['file']}** — "
                        f"{summary['chunks_created']} chunks from "
                        f"{summary['pages_processed']} pages "
                        f"(languages: {', '.join(summary['languages_detected'])})"
                    )
                except Exception as exc:
                    progress_placeholder.error(f"Ingestion failed: {exc}")
                    logger.exception("Ingestion error")
                finally:
                    tmp_path.unlink(missing_ok=True)

    st.divider()

    # --- Retrieval filters ---
    st.subheader("Filters")
    st.multiselect(
        "Document type",
        options=["All", "Contract", "Spec", "RFI"],
        default=st.session_state.filter_doc_types,
        key="filter_doc_types",
        help="Restrict retrieval to selected document types",
    )
    st.radio(
        "Language",
        options=["All", "Arabic only", "English only"],
        index=["All", "Arabic only", "English only"].index(
            st.session_state.filter_language
        ),
        key="filter_language",
        horizontal=False,
    )

    st.divider()

    # --- Stats ---
    if st.session_state.project_id:
        count = vector_store.document_count(st.session_state.project_id)
        st.metric("Indexed chunks", count)

    st.caption(f"Project: **{st.session_state.project_id or 'none'}**")
    st.caption(f"LLM: **{st.session_state.llm_provider}**")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🏗️ Benna AI")
st.caption(
    f"Construction Document Intelligence · GCC · "
    f"Provider: **{st.session_state.llm_provider}**"
    + (f" · Project: **{st.session_state.project_id}**" if st.session_state.project_id else "")
)

if not st.session_state.project_id:
    st.info("👈 Create or select a project in the sidebar, then upload a document to get started.")
    st.stop()

tab1, tab2 = st.tabs(["💬 Chat", "⚡ Conflict Detection"])

# ===========================================================================
# TAB 1 — Chat (unchanged)
# ===========================================================================
with tab1:
    # Replay chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                has_arabic = any("\u0600" <= c <= "\u06FF" for c in msg["content"])
                if has_arabic:
                    st.markdown(
                        f'<div style="direction:rtl;text-align:right">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(msg["content"])
            else:
                rewritten = msg.get("rewritten_query", "")
                original = msg.get("original_query", "")
                if rewritten and rewritten != original:
                    st.markdown(
                        f'<div class="rewritten-query">🔍 Searched as: {rewritten}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        _render_sources(msg["sources"])

    # Chat input
    query_text = st.chat_input("Ask about your documents… / اسأل عن مستنداتك…")

    if query_text:
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in query_text)
        active_filters = _build_filters()

        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            if has_arabic:
                st.markdown(
                    f'<div style="direction:rtl;text-align:right">{query_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(query_text)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            rewritten_placeholder = st.empty()
            collected: List[str] = []
            sources: List[Dict] = []
            rewritten_query = query_text
            error_occurred = False

            try:
                token_gen, sources, rewritten_query = query_stream(
                    query_text=query_text,
                    project_id=st.session_state.project_id,
                    llm_provider=st.session_state.llm_provider,
                    filters=active_filters,
                )

                if rewritten_query and rewritten_query != query_text:
                    rewritten_placeholder.markdown(
                        f'<div class="rewritten-query">🔍 Searched as: {rewritten_query}</div>',
                        unsafe_allow_html=True,
                    )

                for token in token_gen:
                    collected.append(token)
                    answer_placeholder.markdown("".join(collected) + "▌")

                final_answer = "".join(collected)
                answer_placeholder.markdown(final_answer)

            except ConnectionError as exc:
                final_answer = f"⚠️ **Connection error:** {exc}"
                answer_placeholder.error(final_answer)
                error_occurred = True
            except Exception as exc:
                final_answer = f"⚠️ **Error:** {exc}"
                answer_placeholder.error(final_answer)
                logger.exception("Query pipeline error")
                error_occurred = True

            if sources and not error_occurred:
                with st.expander("📎 Sources", expanded=False):
                    _render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer,
                "sources": sources,
                "original_query": query_text,
                "rewritten_query": rewritten_query,
            }
        )

# ===========================================================================
# TAB 2 — Conflict Detection
# ===========================================================================
with tab2:
    st.subheader("Compare two documents for contradictions")
    st.caption(
        "Retrieve relevant clauses independently from two document sets, "
        "then let Benna identify contradictions, alignments, or gaps."
    )

    indexed_files = vector_store.get_indexed_files(st.session_state.project_id)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Document A**")
        doc_a_type = st.selectbox(
            "Filter by type",
            ["contract", "spec", "rfi", "general"],
            key="doc_a_type",
        )
        doc_a_file = st.selectbox(
            "Or specific file (overrides type)",
            ["Any"] + indexed_files,
            key="doc_a_file",
        )
    with col2:
        st.caption("**Document B**")
        doc_b_type = st.selectbox(
            "Filter by type",
            ["spec", "contract", "rfi", "general"],
            key="doc_b_type",
        )
        doc_b_file = st.selectbox(
            "Or specific file (overrides type)",
            ["Any"] + indexed_files,
            key="doc_b_file",
        )

    conflict_query = st.text_area(
        "What topic should Benna compare?",
        placeholder=(
            "e.g. concrete grade requirements, payment terms, "
            "defects liability period, liquidated damages"
        ),
        height=80,
    )

    run_button = st.button("Detect Conflicts", type="primary")

    if run_button and conflict_query.strip():
        doc_a_filter = (
            {"source_file": doc_a_file}
            if doc_a_file != "Any"
            else {"doc_type": doc_a_type}
        )
        doc_b_filter = (
            {"source_file": doc_b_file}
            if doc_b_file != "Any"
            else {"doc_type": doc_b_type}
        )

        with st.spinner("Comparing documents …"):
            try:
                result = detect_conflicts(
                    query=conflict_query,
                    project_id=st.session_state.project_id,
                    doc_a_filter=doc_a_filter,
                    doc_b_filter=doc_b_filter,
                    llm_provider=st.session_state.llm_provider,
                )
            except ConnectionError as exc:
                st.error(f"⚠️ Connection error: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"⚠️ Error: {exc}")
                logger.exception("Conflict detection error")
                st.stop()

        if result["status"] == "insufficient_data":
            st.warning(result["message"])
        else:
            verdict = result.get("verdict", "unclear")

            # Color-coded verdict banner
            if verdict == "contradiction":
                st.error("⚠️ CONTRADICTION DETECTED")
            elif verdict == "aligned":
                st.success("✅ DOCUMENTS ALIGNED")
            elif verdict == "gap":
                st.warning("🔍 GAP IDENTIFIED")
            else:
                st.info("ℹ️ ANALYSIS COMPLETE")

            if result.get("rewritten_query") and result["rewritten_query"] != conflict_query:
                st.markdown(
                    f'<div class="rewritten-query">🔍 Searched as: {result["rewritten_query"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(result["analysis"])

            # Source chunks side by side
            st.divider()
            st.caption("Source chunks used in comparison")
            ca, cb = st.columns(2)

            with ca:
                st.caption("**Document A** sources")
                for chunk in result["chunks_a"]:
                    meta = chunk.get("metadata", {})
                    label = f"{meta.get('source_file', '?')} — p.{meta.get('page_num', '?')}"
                    with st.expander(label):
                        if meta.get("clause_ref"):
                            st.markdown(f"`{meta['clause_ref']}`")
                        if meta.get("section_header"):
                            st.markdown(
                                f'<div class="section-header">§ {meta["section_header"]}</div>',
                                unsafe_allow_html=True,
                            )
                        st.code(chunk["text"][:300], language=None)

            with cb:
                st.caption("**Document B** sources")
                for chunk in result["chunks_b"]:
                    meta = chunk.get("metadata", {})
                    label = f"{meta.get('source_file', '?')} — p.{meta.get('page_num', '?')}"
                    with st.expander(label):
                        if meta.get("clause_ref"):
                            st.markdown(f"`{meta['clause_ref']}`")
                        if meta.get("section_header"):
                            st.markdown(
                                f'<div class="section-header">§ {meta["section_header"]}</div>',
                                unsafe_allow_html=True,
                            )
                        st.code(chunk["text"][:300], language=None)

    elif run_button and not conflict_query.strip():
        st.warning("Please enter a topic to compare.")
