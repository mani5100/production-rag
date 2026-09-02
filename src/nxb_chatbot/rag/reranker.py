from langchain_community.document_compressors import FlashrankRerank

FlashrankRerank.model_rebuild()

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
)
from langchain_ollama import ChatOllama

from nxb_chatbot.core.config import settings


def get_reranking_retriever(
    base_retriever: VectorStoreRetriever,
) -> ContextualCompressionRetriever:
    """
    Wraps a base Qdrant retriever with FlashrankRerank compressor.

    Flow:
        base_retriever fetches TOP_K candidates from Qdrant
            ↓
        FlashrankRerank re-scores all candidates against query
            ↓
        Returns TOP_N most relevant documents
    """
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=10,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def get_multi_query_retriever(
    base_retriever: VectorStoreRetriever,
) -> MultiQueryRetriever:
    """
    Wraps a base Qdrant retriever with multi-query expansion.

    Flow:
        The original query always runs verbatim (include_original=True),
        plus the LLM generates 3-5 rephrasings of it
            ↓
        base_retriever runs once per query/rephrasing
            ↓
        Results are merged and deduplicated

    Query generation uses a dedicated low-temperature, seeded LLM instance
    instead of the shared `llm` (temperature=LLM_TEMPERATURE, used for
    generation/reflection/grading) so the candidate pool handed to the
    reranker is reproducible across runs rather than depending on sampling
    variance in the paraphrases.
    """
    expansion_llm = ChatOllama(
        model=settings.LLM_MODEL,
        temperature=settings.QUERY_EXPANSION_TEMPERATURE,
        base_url=settings.OLLAMA_BASE_URL,
        seed=settings.QUERY_EXPANSION_SEED,
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=expansion_llm,
        include_original=True,
    )
