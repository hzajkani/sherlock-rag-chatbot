"""Command-line entry point for quick local testing."""

from __future__ import annotations

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.chat_engine import build_chat_engine
from src.model_loader import get_embedding_model, initialise_llm


def main_chat_loop() -> None:
    """Initialise models and drop the user into LlamaIndex's chat REPL."""
    print("--- Initialising models... ---")
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: BaseChatEngine = build_chat_engine(llm, embed_model)
    print("--- Sherlock Holmes RAG Chatbot ready. Type 'exit' to quit. ---")
    chat_engine.chat_repl()
