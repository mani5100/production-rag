import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

from nxb_chatbot.vector_store.qdrant_client import init_collection
from nxb_chatbot.ingestion.pipeline import run_ingestion_pipeline
from nxb_chatbot.vector_store.qdrant_client import get_qdrant_client
from nxb_chatbot.core.config import settings


def test_ingestion():
    print("\n" + "="*60)
    print("STEP 1 — Initializing Qdrant collection")
    print("="*60)
    init_collection()

    print("\n" + "="*60)
    print("STEP 2 — Running ingestion pipeline")
    print("="*60)
    run_ingestion_pipeline()

    print("\n" + "="*60)
    print("STEP 3 — Verifying chunks in Qdrant")
    print("="*60)
    client = get_qdrant_client()

    collection_info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
    total_chunks = collection_info.points_count
    print(f"Total chunks in Qdrant: {total_chunks}")

    # Scroll and check table chunks
    results, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "metadata.has_table",
                    "match": {"value": True},
                }
            ]
        },
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    print(f"Table chunks found: {len(results)}")

    if results:
        print("\n--- Sample Table Chunk ---")
        sample = results[0]
        meta = sample.payload.get("metadata", {})
        print(f"File    : {meta.get('file_name')}")
        print(f"Page    : {meta.get('page')}")
        print(f"Hash    : {meta.get('file_hash')}")
        print(f"Content preview:\n{sample.payload.get('page_content', '')[:500]}")

    # Scroll and check text chunks
    text_results, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "metadata.has_table",
                    "match": {"value": False},
                }
            ]
        },
        limit=3,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\nText chunks sample (first 3):")
    for r in text_results:
        meta = r.payload.get("metadata", {})
        print(f"  - {meta.get('file_name')} | page {meta.get('page')} | {len(r.payload.get('page_content', ''))} chars")

    print("\n✅ Ingestion test complete.")


if __name__ == "__main__":
    test_ingestion()