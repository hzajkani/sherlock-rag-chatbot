"""Reusable Streamlit components for the chatbot UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    AVAILABLE_LLM_MODELS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_TOP_K,
    USE_HYDE_DEFAULT,
    USE_RERANKER_DEFAULT,
)
from src.prompts import DEFAULT_PROMPT_KEY, PROMPTS


@dataclass(frozen=True)
class SidebarState:
    """Snapshot of all user-tunable options from the sidebar."""

    persona_key: str
    system_prompt: str
    model_name: str
    temperature: float
    similarity_top_k: int
    use_reranker: bool
    use_hyde: bool

    def fingerprint(self) -> tuple[Any, ...]:
        """Stable tuple used to detect sidebar changes between reruns."""
        return (
            self.persona_key,
            self.model_name,
            self.temperature,
            self.similarity_top_k,
            self.use_reranker,
            self.use_hyde,
        )


def _reset_defaults() -> None:
    """Reset every tunable control to its value in ``src/config.py``."""
    st.session_state["persona_key"] = DEFAULT_PROMPT_KEY
    st.session_state["model_name"] = LLM_MODEL
    st.session_state["temperature"] = LLM_TEMPERATURE
    st.session_state["similarity_top_k"] = SIMILARITY_TOP_K
    st.session_state["use_reranker"] = USE_RERANKER_DEFAULT
    st.session_state["use_hyde"] = USE_HYDE_DEFAULT


def render_sidebar(on_new_chat) -> SidebarState:
    """Render the configuration sidebar and return the current state."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        if st.button("🔄 New Chat", use_container_width=True, type="primary"):
            on_new_chat()

        st.divider()
        st.subheader("Persona")
        persona_key = st.selectbox(
            "System prompt",
            options=list(PROMPTS.keys()),
            key="persona_key",
            help="Controls the voice and style of the chatbot's answers.",
        )

        st.subheader("Model")
        model_name = st.selectbox(
            "LLM",
            options=AVAILABLE_LLM_MODELS,
            key="model_name",
            help="Any Groq-hosted model listed in src/config.py.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="temperature",
            help="Lower values make the chatbot more deterministic.",
        )

        st.subheader("Retrieval")
        similarity_top_k = st.slider(
            "Similarity top-k",
            min_value=1,
            max_value=10,
            key="similarity_top_k",
            help="How many passages to retrieve before (optionally) reranking.",
        )
        use_reranker = st.toggle(
            "Enable reranker",
            key="use_reranker",
            help=(
                "Apply a cross-encoder to re-order retrieved passages by "
                "true relevance before sending them to the LLM."
            ),
        )
        use_hyde = st.toggle(
            "Enable HyDE query rewriting",
            key="use_hyde",
            help=(
                "Generate a hypothetical answer first and use it as the "
                "retrieval query (LlamaIndex's HyDEQueryTransform)."
            ),
        )

        st.divider()
        st.button(
            "↩️ Reset to defaults",
            on_click=_reset_defaults,
            use_container_width=True,
        )

        st.caption(
            "Built with LlamaIndex + Groq + Streamlit. Source texts are "
            "public-domain works from Project Gutenberg."
        )

    return SidebarState(
        persona_key=persona_key,
        system_prompt=PROMPTS[persona_key],
        model_name=model_name,
        temperature=temperature,
        similarity_top_k=similarity_top_k,
        use_reranker=use_reranker,
        use_hyde=use_hyde,
    )


def render_sources(source_nodes: list | None) -> None:
    """Render retrieved source chunks inside a collapsed expander."""
    if not source_nodes:
        return

    with st.expander(f"📚 Sources ({len(source_nodes)})", expanded=False):
        for i, node in enumerate(source_nodes, start=1):
            metadata = getattr(node, "metadata", {}) or {}
            raw_name = metadata.get("file_name") or metadata.get("filename", "")
            title = Path(raw_name).stem if raw_name else f"Passage {i}"
            score = getattr(node, "score", None)
            score_badge = f" · similarity **{score:.3f}**" if score is not None else ""

            st.markdown(f"**{i}. {title}**{score_badge}")
            text = node.get_content().strip().replace("\n\n", "\n")
            st.markdown(f"> {text}")
            if i < len(source_nodes):
                st.divider()


def render_feedback_buttons(turn_id: str) -> None:
    """Render thumbs-up/thumbs-down buttons for one assistant turn.

    Feedback is only persisted in session state — useful for a later
    analytics hook but deliberately not wired to a backend yet.
    """
    feedback: dict[str, str] = st.session_state.setdefault("feedback", {})
    current = feedback.get(turn_id)

    cols = st.columns([1, 1, 14])
    with cols[0]:
        if st.button(
            "👍",
            key=f"up_{turn_id}",
            type="primary" if current == "up" else "secondary",
            help="Mark this answer as helpful.",
        ):
            feedback[turn_id] = "up"
            st.rerun()
    with cols[1]:
        if st.button(
            "👎",
            key=f"down_{turn_id}",
            type="primary" if current == "down" else "secondary",
            help="Mark this answer as unhelpful.",
        ):
            feedback[turn_id] = "down"
            st.rerun()
