"""Vector index construction and persistence.

The index is built once from the files under ``data/`` and persisted under
``local_storage/vector_store/``. Subsequent runs load from disk.
"""

from __future__ import annotations

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    VECTOR_STORE_PATH,
)


def _build_new_index(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Read the source texts, split them, embed them, and persist the index."""
    print(f"Building new vector store from files in '{DATA_PATH}'...")

    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH.as_posix(),
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. "
            "Add the Sherlock Holmes text files before building the index."
        )

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
    )

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def build_or_load_index(
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Load the persisted index if present, otherwise build and persist it."""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix(),
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )
    return _build_new_index(embed_model)
