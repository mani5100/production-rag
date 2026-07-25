import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from nxb_chatbot.core.config import settings
from nxb_chatbot.core.embeddings import get_dense_embedder

logger = logging.getLogger(__name__)

SPARSE_MODEL = "Qdrant/bm25"


def get_qdrant_client() -> QdrantClient:
    """Returns a raw Qdrant client instance."""
    return QdrantClient(url=settings.QDRANT_URL)


def init_collection() -> None:
    """
    One-time collection initialization.
    Creates the collection with dense + sparse vector configs.
    Run this manually once before ingestion.
    """
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION_NAME in existing:
        logger.info(
            f"Collection '{settings.QDRANT_COLLECTION_NAME}' already exists. Skipping."
        )
        return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )

    logger.info(
        f"Collection '{settings.QDRANT_COLLECTION_NAME}' created "
        f"with dense (1536, Cosine) + sparse (BM25) config."
    )


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_existing_hash(client: QdrantClient, file_name: str) -> str | None:
    """
    Query Qdrant metadata to find the stored hash for a given file.
    Returns None if file was never ingested.
    """
    results, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "metadata.file_name",
                    "match": {"value": file_name},
                }
            ]
        },
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        return None

    return results[0].payload.get("metadata", {}).get("file_hash")


def delete_file_chunks(client: QdrantClient, file_name: str) -> None:
    """Delete all chunks belonging to a specific file."""
    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector={
            "filter": {
                "must": [
                    {
                        "key": "metadata.file_name",
                        "match": {"value": file_name},
                    }
                ]
            }
        },
    )
    logger.info(f"Deleted existing chunks for: {file_name}")


def upsert_documents(documents: list[Document], file_path: Path) -> None:
    """
    Upsert documents into Qdrant with duplicate + change detection.

    - Same file, same hash  → skip
    - Same file, new hash   → delete old chunks, upsert fresh
    - New file              → upsert directly
    """
    client = get_qdrant_client()
    file_name = file_path.name
    file_hash = compute_file_hash(file_path)

    existing_hash = get_existing_hash(client, file_name)

    if existing_hash == file_hash:
        logger.info(f"Skipping '{file_name}' — no changes detected.")
        return

    if existing_hash is not None:
        logger.info(f"Changes detected in '{file_name}' — re-ingesting.")
        delete_file_chunks(client, file_name)
    else:
        logger.info(f"New file detected: '{file_name}' — ingesting.")

    # Attach file hash to every chunk metadata
    for doc in documents:
        doc.metadata["file_hash"] = file_hash

    vector_store = get_vector_store(client)
    vector_store.add_documents(documents)

    logger.info(f"Upserted {len(documents)} chunks for '{file_name}'.")


def get_vector_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    """
    Returns a QdrantVectorStore instance configured for hybrid search.
    Used by both ingestion and the RAG retriever.
    """
    if client is None:
        client = get_qdrant_client()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=get_dense_embedder(),
        sparse_embedding=FastEmbedSparse(model_name=SPARSE_MODEL),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )