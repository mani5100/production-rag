import asyncio
import logging
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

from langchain_core.messages import HumanMessage
from nxb_chatbot.rag.graph import get_compiled_graph


async def test_rag():
    print("\n" + "="*60)
    print("STEP 1 — Compiling RAG graph")
    print("="*60)
    graph, pool = await get_compiled_graph()
    print("Graph compiled successfully.")

    # thread_id simulates a user session
    config = {"configurable": {"thread_id": "test-session-001"}}

    # -------------------------------------------------------
    # Turn 1 — First question (no reformulation expected)
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2 — Turn 1 (first question)")
    print("="*60)

    question_1 = "What are the benefits that are given to nextbridge employees"

    print(f"Question: {question_1}\n")

    state_1 = await graph.ainvoke(
        {"messages": [HumanMessage(content=question_1)],
         "retrieved_docs": [],
         "retrieval_filters": None,
         "standalone_query": None,
         },
        config=config,
    )

    print(f"Standalone query : {state_1['standalone_query']}")
    print(f"Chunks retrieved : {len(state_1['retrieved_docs'])}")
    print("\nRetrieved chunks:")
    for i, doc in enumerate(state_1["retrieved_docs"], 1):
        meta = doc["metadata"]          # ← dict key, not attribute
        print(
            f"  [{i}] {meta.get('file_name')} | "
            f"page {meta.get('page')} | "
            f"table: {meta.get('has_table')} | "
            f"rerank_score: {meta.get('relevance_score', 'N/A')}"
        )

    print(f"\nAnswer:\n{state_1['messages'][-1].content}")

    # -------------------------------------------------------
    # Turn 2 — Follow-up (reformulation expected)
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3 — Turn 2 (follow-up question)")
    print("="*60)

    question_2 = "I am Associate Project Manager what will i get?"

    print(f"Question: {question_2}\n")

    state_2 = await graph.ainvoke(
        {"messages": [HumanMessage(content=question_2)]},
        config=config,
    )

    print(f"Standalone query  : {state_2['standalone_query']}")
    print(f"Chunks retrieved  : {len(state_2['retrieved_docs'])}")
    print(f"\nAnswer:\n{state_2['messages'][-1].content}")

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------
    await pool.close()
    print("\n✅ RAG test complete.")


if __name__ == "__main__":
    asyncio.run(test_rag())