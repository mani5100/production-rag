import logging
from typing import Any

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize.base import BaseVectorizer

from nxb_chatbot.core.config import settings
from nxb_chatbot.core.embeddings import get_dense_embedder

logger = logging.getLogger(__name__)


class OllamaVectorizer(BaseVectorizer):
    """
    Adapts the project's Ollama embedding model (get_dense_embedder)
    to redisvl's vectorizer interface, so the semantic cache embeds
    queries with the same model used everywhere else in the pipeline.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model=settings.EMBEDDING_MODEL,
            dims=settings.EMBEDDING_DIMENSIONS,
            **kwargs,
        )
        self._embedder = get_dense_embedder()

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_many(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._embedder.embed_documents(texts)


_semantic_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache | None:
    """
    Returns a singleton SemanticCache instance, or None if disabled.

    Scoped ONLY to the knowledge-lookup path (query_reformulator ->
    ... -> reflect_answer "pass"). Never call this for the stateful
    transactional nodes (meal_subscription, mis_request,
    employee_request, and their status-check counterparts) or for
    conversational_response.
    """
    global _semantic_cache

    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    if _semantic_cache is None:
        logger.info(
            "Initializing semantic cache: index=%s threshold=%s ttl=%ss",
            settings.SEMANTIC_CACHE_INDEX_NAME,
            settings.SEMANTIC_CACHE_THRESHOLD,
            settings.SEMANTIC_CACHE_TTL,
        )

        _semantic_cache = SemanticCache(
            name=settings.SEMANTIC_CACHE_INDEX_NAME,
            redis_url=settings.REDIS_URL,
            vectorizer=OllamaVectorizer(),
            distance_threshold=1 - settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
        )

    return _semantic_cache


def check_semantic_cache(query: str) -> dict | None:
    """
    Looks up `query` in the semantic cache.

    Returns the cached entry dict (with "response" and "metadata" keys)
    on a hit, or None on a miss or when the cache is disabled.
    """
    cache = get_semantic_cache()

    if cache is None:
        return None

    results = cache.check(prompt=query, num_results=1)

    if not results:
        return None

    hit = results[0]

    logger.info(
        "Semantic cache HIT (distance=%.4f) for query: %s",
        hit.get("vector_distance", -1),
        query,
    )

    return hit


def store_semantic_cache(query: str, answer: str, retrieved_docs: list[dict]) -> None:
    """
    Writes an approved answer to the semantic cache.

    Only call this after the answer has passed Self-RAG reflection
    (reflection_action == "pass") and web_search_used is False.
    """
    cache = get_semantic_cache()

    if cache is None:
        return

    cache.store(
        prompt=query,
        response=answer,
        metadata={
            "doc_count": len(retrieved_docs),
            "source_files": list(
                {
                    doc.get("metadata", {}).get("file_name")
                    for doc in retrieved_docs
                    if doc.get("metadata", {}).get("file_name")
                }
            ),
        },
    )

    logger.info("Semantic cache STORE for query: %s", query)