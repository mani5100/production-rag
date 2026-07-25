from langchain_community.document_compressors import FlashrankRerank

FlashrankRerank.model_rebuild()

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever

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
        top_n=settings.RERANKER_TOP_N,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )