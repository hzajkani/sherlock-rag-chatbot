"""Runtime configuration for the Sherlock Holmes RAG chatbot.

All tunable defaults live here so that both the Streamlit UI and the
evaluation harness import one source of truth. Paths use ``pathlib`` so the
project is portable across machines.
"""

from __future__ import annotations

from pathlib import Path

from src.prompts import CANON_EXPERT

# --- LLM Configuration ---
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_NEW_TOKENS: int = 768

# Models exposed in the UI model selector. All must be available on Groq.
AVAILABLE_LLM_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# Default system prompt (the "Canon Expert" persona).
LLM_SYSTEM_PROMPT: str = CANON_EXPERT

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG / Vector Store Configuration ---
SIMILARITY_TOP_K: int = 4
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# --- Reranker Configuration ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N: int = 3
USE_RERANKER_DEFAULT: bool = False

# --- Query Rewriting (HyDE) ---
USE_HYDE_DEFAULT: bool = False

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Persistent Storage Paths ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage" / "vector_store"
