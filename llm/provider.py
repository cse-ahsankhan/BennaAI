from __future__ import annotations

import logging
from typing import List, Dict, Any, Generator

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Benna AI, a construction document assistant specializing in GCC projects. "
    "Answer questions based strictly on the provided document context. "
    "Always cite the source document and clause/section reference. "
    "If the answer is not in the context, say so clearly. "
    "Support both Arabic and English queries."
)


def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "unknown")
        page = meta.get("page_num", "?")
        clause = meta.get("clause_ref", "")
        clause_str = f" | Clause: {clause}" if clause else ""
        parts.append(
            f"[{i}] Source: {source} | Page: {page}{clause_str}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    return (
        f"Context from project documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

class _ClaudeProvider:
    def __init__(self) -> None:
        import anthropic

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = "claude-sonnet-4-6"

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context = _format_context(context_chunks)
        user_content = _build_prompt(query, context)

        full_response = []
        with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                full_response.append(text)

        return "".join(full_response)

    def generate_stream(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        context = _format_context(context_chunks)
        user_content = _build_prompt(query, context)

        with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                yield text


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

class _OllamaProvider:
    def __init__(self) -> None:
        try:
            from langchain_community.llms import Ollama

            self._llm = Ollama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_MODEL,
            )
        except ImportError:
            raise ImportError("langchain-community is required for Ollama support")

    def _full_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context = _format_context(context_chunks)
        return (
            f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            + _build_prompt(query, context)
        )

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        prompt = self._full_prompt(query, context_chunks)
        try:
            return self._llm.invoke(prompt)
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                    "Is the Ollama server running? Start it with: ollama serve"
                ) from exc
            raise

    def generate_stream(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        prompt = self._full_prompt(query, context_chunks)
        try:
            for chunk in self._llm.stream(prompt):
                yield chunk
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                    "Is the Ollama server running? Start it with: ollama serve"
                ) from exc
            raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_cache: Dict[str, Any] = {}


def get_llm(provider: str | None = None):
    """
    Return the LLM provider instance based on LLM_PROVIDER config (or override).
    Instances are cached per provider name.
    """
    name = (provider or config.LLM_PROVIDER).lower()

    if name not in _provider_cache:
        if name == "claude":
            logger.info("Initialising Claude provider (model: claude-sonnet-4-6)")
            _provider_cache[name] = _ClaudeProvider()
        elif name == "ollama":
            logger.info(
                "Initialising Ollama provider (model: %s @ %s)",
                config.OLLAMA_MODEL,
                config.OLLAMA_BASE_URL,
            )
            _provider_cache[name] = _OllamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: '{name}'. Choose 'ollama' or 'claude'.")

    return _provider_cache[name]


def clear_cache() -> None:
    """Clear provider cache (useful when switching providers at runtime)."""
    _provider_cache.clear()
