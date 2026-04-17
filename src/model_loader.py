"""Factory functions for the LLM and embedding model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.config import (
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

load_dotenv()


def initialise_llm(
    model_name: str | None = None,
    temperature: float | None = None,
) -> Groq:
    """Return a configured Groq LLM client.

    Both arguments are optional so the UI can override the defaults at
    runtime without the CLI/evaluation paths having to care.
    """
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file "
            "(see .env.example for the format)."
        )

    return Groq(
        api_key=api_key,
        model=model_name or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
    )


def get_embedding_model() -> HuggingFaceEmbedding:
    """Return the HuggingFace embedding model, cached on disk."""
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )
