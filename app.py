"""Streamlit front-end for the Sherlock Holmes RAG chatbot."""

from __future__ import annotations

import uuid

import streamlit as st
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.chat_engine import build_chat_engine
from src.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_TOP_K,
    USE_HYDE_DEFAULT,
    USE_RERANKER_DEFAULT,
)
from src.model_loader import get_embedding_model, initialise_llm
from src.prompts import DEFAULT_PROMPT_KEY
from ui.components import (
    SidebarState,
    render_feedback_buttons,
    render_sidebar,
    render_sources,
)
from ui.sample_questions import SAMPLE_QUESTIONS

st.set_page_config(
    page_title="Sherlock Holmes — Canon Chatbot",
    page_icon="🔍",
    layout="centered",
)


# --- Resource caches -------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> HuggingFaceEmbedding:
    return get_embedding_model()


@st.cache_resource(show_spinner="Connecting to the LLM...")
def load_llm(model_name: str, temperature: float) -> Groq:
    return initialise_llm(model_name=model_name, temperature=temperature)


# --- Session state helpers -------------------------------------------------


def _init_session_state() -> None:
    """Initialise all session-state keys to a clean first-visit state."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("chat_engine", None)
    st.session_state.setdefault("engine_fingerprint", None)

    # Pre-seed sidebar widget defaults so the controls open on the values
    # defined in src/config.py.
    st.session_state.setdefault("persona_key", DEFAULT_PROMPT_KEY)
    st.session_state.setdefault("model_name", LLM_MODEL)
    st.session_state.setdefault("temperature", LLM_TEMPERATURE)
    st.session_state.setdefault("similarity_top_k", SIMILARITY_TOP_K)
    st.session_state.setdefault("use_reranker", USE_RERANKER_DEFAULT)
    st.session_state.setdefault("use_hyde", USE_HYDE_DEFAULT)


def _reset_chat() -> None:
    """Clear the transcript and force a fresh chat engine on the next run."""
    st.session_state["messages"] = []
    st.session_state["feedback"] = {}
    st.session_state["pending_question"] = None
    st.session_state["chat_engine"] = None
    st.session_state["engine_fingerprint"] = None


def _get_chat_engine(sidebar: SidebarState) -> BaseChatEngine:
    """Return the chat engine, rebuilding it when a sidebar control changes."""
    fingerprint = sidebar.fingerprint()
    cached_engine = st.session_state.get("chat_engine")
    cached_fp = st.session_state.get("engine_fingerprint")
    if cached_engine is not None and cached_fp == fingerprint:
        return cached_engine

    llm = load_llm(sidebar.model_name, sidebar.temperature)
    embed_model = load_embedder()

    engine = build_chat_engine(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=sidebar.similarity_top_k,
        system_prompt=sidebar.system_prompt,
        use_reranker=sidebar.use_reranker,
        use_hyde=sidebar.use_hyde,
    )
    st.session_state["chat_engine"] = engine
    st.session_state["engine_fingerprint"] = fingerprint
    return engine


# --- Rendering -------------------------------------------------------------


def _render_header() -> None:
    st.title("🔍 Sherlock Holmes — Canon Chatbot")
    st.caption(
        "A Retrieval-Augmented Generation chatbot grounded in "
        "*The Adventures of Sherlock Holmes* and *The Memoirs of Sherlock "
        "Holmes* by Sir Arthur Conan Doyle (public domain via Project "
        "Gutenberg)."
    )


def _render_sample_questions() -> None:
    st.markdown("#### Try one of these to get started:")
    columns = st.columns(2)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        column = columns[i % 2]
        if column.button(
            question,
            key=f"sample_{i}",
            use_container_width=True,
        ):
            st.session_state["pending_question"] = question
            st.rerun()


def _render_transcript() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))
                render_feedback_buttons(message["turn_id"])


def _answer_question(chat_engine: BaseChatEngine, question: str) -> None:
    """Append the user question, generate an answer, and store it."""
    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the canon..."):
            response = chat_engine.chat(question)
        answer = str(response)
        st.markdown(answer)
        source_nodes = getattr(response, "source_nodes", []) or []
        render_sources(source_nodes)
        turn_id = uuid.uuid4().hex
        render_feedback_buttons(turn_id)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_nodes,
            "turn_id": turn_id,
        }
    )


# --- Main ------------------------------------------------------------------


def main() -> None:
    _init_session_state()
    sidebar = render_sidebar(on_new_chat=_reset_chat)
    _render_header()

    chat_engine = _get_chat_engine(sidebar)

    if not st.session_state["messages"]:
        _render_sample_questions()
    else:
        _render_transcript()

    pending = st.session_state.get("pending_question")
    if pending:
        st.session_state["pending_question"] = None
        _answer_question(chat_engine, pending)
        st.rerun()

    user_input = st.chat_input("Ask about any character, case, or clue...")
    if user_input:
        _answer_question(chat_engine, user_input)
        st.rerun()


if __name__ == "__main__":
    main()
