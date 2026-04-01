"""
Benna AI — Construction Document Intelligence
Streamlit UI
"""
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when running from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from retrieval import vector_store  # noqa: E402
from pipeline.ingest_pipeline import ingest_document  # noqa: E402
from pipeline.query_pipeline import query_stream  # noqa: E402
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
# CSS — RTL support + chat bubbles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .rtl-input textarea {
        direction: rtl;
        text-align: right;
    }
    .chat-user {
        background: #DCF8C6;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        text-align: right;
        direction: rtl;
    }
    .chat-assistant {
        background: #F1F0F0;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .source-badge {
        display: inline-block;
        background: #E8F4FD;
        border: 1px solid #BDD7EE;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82em;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = config.LLM_PROVIDER
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "embedding_warned" not in st.session_state:
    st.session_state.embedding_warned = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/skyscraper.png", width=64)
    st.title("Benna AI 🏗️")
    st.caption("Construction Document Intelligence for the GCC")

    st.divider()

    # --- LLM provider toggle ---
    st.subheader("LLM Provider")
    provider_choice = st.radio(
        "Select LLM",
        options=["ollama", "claude"],
        index=0 if st.session_state.llm_provider == "ollama" else 1,
        horizontal=True,
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
        # Refresh project list
        existing_projects = vector_store.list_projects()

    if existing_projects:
        selected = st.selectbox(
            "Or select existing project",
            options=["— select —"] + existing_projects,
        )
        if selected != "— select —":
            if selected != st.session_state.project_id:
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
            messages_log = []

            def _progress(msg: str) -> None:
                messages_log.append(msg)
                progress_placeholder.info("\n\n".join(messages_log[-3:]))

            with st.spinner("Ingesting document …"):
                try:
                    if not st.session_state.embedding_warned:
                        st.toast(
                            "First run: downloading embedding model (~1.1 GB). This may take a few minutes.",
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

    # --- Indexed doc count ---
    if st.session_state.project_id:
        count = vector_store.document_count(st.session_state.project_id)
        st.metric("Indexed chunks", count)

    st.divider()
    st.caption(f"Active project: **{st.session_state.project_id or 'none'}**")
    st.caption(f"LLM: **{st.session_state.llm_provider}**")


# ---------------------------------------------------------------------------
# Main area
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

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    clause = f" · Clause {src['clause_ref']}" if src["clause_ref"] else ""
                    origins = " + ".join(sorted(set(src.get("retrieval_sources", []))))
                    st.markdown(
                        f"**{src['file']}** · Page {src['page']}{clause} "
                        f"· RRF: `{src['rrf_score']}` · via `{origins}`"
                    )
                    if src.get("text_snippet"):
                        st.caption(f"> {src['text_snippet']} …")

# Chat input
query_text = st.chat_input(
    "Ask about your documents… / اسأل عن مستنداتك…",
)

if query_text:
    # Detect RTL and display accordingly
    has_arabic = any("\u0600" <= c <= "\u06FF" for c in query_text)
    display_query = query_text

    st.session_state.messages.append({"role": "user", "content": display_query})
    with st.chat_message("user"):
        if has_arabic:
            st.markdown(
                f'<div style="direction:rtl;text-align:right">{display_query}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(display_query)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        collected = []
        sources = []
        error_occurred = False

        try:
            token_gen, sources = query_stream(
                query_text=query_text,
                project_id=st.session_state.project_id,
                llm_provider=st.session_state.llm_provider,
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
                for src in sources:
                    clause = f" · Clause {src['clause_ref']}" if src["clause_ref"] else ""
                    origins = " + ".join(sorted(set(src.get("retrieval_sources", []))))
                    st.markdown(
                        f"**{src['file']}** · Page {src['page']}{clause} "
                        f"· RRF: `{src['rrf_score']}` · via `{origins}`"
                    )
                    if src.get("text_snippet"):
                        st.caption(f"> {src['text_snippet']} …")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "sources": sources,
        }
    )
