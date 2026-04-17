"""Chat engine factory.

Builds a fresh LlamaIndex ``ContextChatEngine`` per call. Each call gets its
own ``ChatMemoryBuffer`` so multi-session UIs (Streamlit) do not accidentally
share conversation history between users or between "New Chat" resets.

Supports two optional retrieval-time enhancements the project studied in
class:

* **Reranker** – a cross-encoder (``SentenceTransformerRerank``) applied as a
  node post-processor to re-order initial retrieval results by true
  query-document relevance before they reach the LLM.
* **HyDE** – ``HyDEQueryTransform`` generates a hypothetical answer that is
  embedded and used as the retrieval query, which often surfaces better
  passages when the user's question is short or vague.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import TransformRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.config import (
    CHAT_MEMORY_TOKEN_LIMIT,
    LLM_SYSTEM_PROMPT,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    SIMILARITY_TOP_K,
    USE_HYDE_DEFAULT,
    USE_RERANKER_DEFAULT,
)
from src.indexing import build_or_load_index


def build_chat_engine(
    llm: Groq,
    embed_model: HuggingFaceEmbedding,
    *,
    similarity_top_k: int = SIMILARITY_TOP_K,
    system_prompt: str = LLM_SYSTEM_PROMPT,
    use_reranker: bool = USE_RERANKER_DEFAULT,
    use_hyde: bool = USE_HYDE_DEFAULT,
) -> BaseChatEngine:
    """Return a chat engine wired up for the requested retrieval strategy."""
    index: VectorStoreIndex = build_or_load_index(embed_model)

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    if use_hyde:
        retriever = TransformRetriever(
            retriever=retriever,
            query_transform=HyDEQueryTransform(llm=llm, include_original=True),
        )

    node_postprocessors = []
    if use_reranker:
        node_postprocessors.append(
            SentenceTransformerRerank(
                top_n=RERANKER_TOP_N,
                model=RERANKER_MODEL_NAME,
            )
        )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT,
    )

    return ContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=system_prompt,
        node_postprocessors=node_postprocessors,
    )
