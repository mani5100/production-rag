import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.nxb_chatbot.evaluation.dataset import load_golden_dataset
from src.nxb_chatbot.evaluation.models import EvaluationRun, GoldenSample

from nxb_chatbot.rag.graph import get_compiled_graph

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


DEFAULT_DATASET = "data/golden_dataset/golden_dataset.json"
DEFAULT_OUTPUT = "evaluation_results/raw_runs.json"


def extract_contexts(
    retrieved_docs: list[dict[str, Any]] | None,
) -> list[str]:
    if not retrieved_docs:
        return []

    contexts: list[str] = []

    for doc in retrieved_docs:
        if isinstance(doc, dict):
            content = doc.get("page_content")

            if content:
                contexts.append(str(content))

        elif hasattr(doc, "page_content"):
            contexts.append(str(doc.page_content))

    return contexts


def normalize_documents(
    retrieved_docs: list[Any] | None,
) -> list[dict[str, Any]]:
    if not retrieved_docs:
        return []

    normalized: list[dict[str, Any]] = []

    for doc in retrieved_docs:
        if isinstance(doc, dict):
            normalized.append(
                {
                    "page_content": doc.get(
                        "page_content",
                        "",
                    ),
                    "metadata": doc.get(
                        "metadata",
                        {},
                    ),
                }
            )

        elif hasattr(doc, "page_content"):
            normalized.append(
                {
                    "page_content": (doc.page_content),
                    "metadata": getattr(
                        doc,
                        "metadata",
                        {},
                    ),
                }
            )

        else:
            normalized.append(
                {
                    "page_content": str(doc),
                    "metadata": {},
                }
            )

    return normalized


def extract_generated_answer(
    state: dict[str, Any],
) -> str:

    answer = state.get("generated_answer")

    if answer:
        return str(answer)

    messages = state.get(
        "messages",
        [],
    )

    if messages:
        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            None,
        )

        if content:
            return str(content)

    return ""


def build_initial_state(
    sample: GoldenSample,
) -> dict[str, Any]:

    return {
        "messages": [HumanMessage(content=sample.question)],
        "retrieved_docs": [],
        "standalone_query": "",
        "retrieval_queries": [],
        "generated_answer": "",
        "web_search_used": False,
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "reflection_attempts": 0,
        "grade_verdict": None,
        "grade_reason": None,
        "reflection_action": None,
        "reflection_reason": None,
        "reflection_feedback": None,
    }


def state_to_evaluation_run(
    sample: GoldenSample,
    state: dict[str, Any],
) -> EvaluationRun:

    retrieved_docs = state.get(
        "retrieved_docs",
        [],
    )

    return EvaluationRun(
        id=sample.id,
        question=sample.question,
        expected_answer=(sample.expected_answer),
        category=sample.category,
        generated_answer=(extract_generated_answer(state)),
        retrieved_contexts=(extract_contexts(retrieved_docs)),
        retrieved_documents=(normalize_documents(retrieved_docs)),
        expected_source_doc=(sample.source_doc),
        expected_source_page=(sample.source_page),
        standalone_query=state.get("standalone_query"),
        retrieval_queries=state.get(
            "retrieval_queries",
            [],
        )
        or [],
        web_search_used=bool(
            state.get(
                "web_search_used",
                False,
            )
        ),
        retrieval_attempts=int(
            state.get(
                "retrieval_attempts",
                0,
            )
            or 0
        ),
        generation_attempts=int(
            state.get(
                "generation_attempts",
                0,
            )
            or 0
        ),
        reflection_attempts=int(
            state.get(
                "reflection_attempts",
                0,
            )
            or 0
        ),
        grade_verdict=state.get("grade_verdict"),
        grade_reason=state.get("grade_reason"),
        reflection_action=state.get("reflection_action"),
        reflection_reason=state.get("reflection_reason"),
    )


async def run_sample(
    sample: GoldenSample,
    graph: Any,
) -> EvaluationRun:

    print()
    print("=" * 70)
    print(f"Running {sample.id}: " f"{sample.question}")
    print("=" * 70)

    try:
        initial_state = build_initial_state(sample)

        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": f"eval-{sample.id}"}},
        )

        result = state_to_evaluation_run(
            sample=sample,
            state=final_state,
        )

        print(f"[SUCCESS] {sample.id}")

        print(
            "  Retrieved contexts:",
            len(result.retrieved_contexts),
        )

        print(
            "  Retrieval attempts:",
            result.retrieval_attempts,
        )

        print(
            "  Generation attempts:",
            result.generation_attempts,
        )

        print(
            "  Reflection attempts:",
            result.reflection_attempts,
        )

        print(
            "  Web search:",
            result.web_search_used,
        )

        return result

    except Exception as exc:
        print(f"[ERROR] {sample.id}: " f"{exc}")

        traceback.print_exc()

        return EvaluationRun(
            id=sample.id,
            question=sample.question,
            expected_answer=(sample.expected_answer),
            category=sample.category,
            expected_source_doc=(sample.source_doc),
            expected_source_page=(sample.source_page),
            status="error",
            error=str(exc),
        )


def save_results(
    results: list[EvaluationRun],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = [result.to_dict() for result in results]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"Saved {len(results)} runs " f"to {output_path}")


def select_samples(
    samples: list[GoldenSample],
    ids: list[str] | None,
) -> list[GoldenSample]:

    if not ids:
        return samples

    requested_ids = set(ids)

    selected = [sample for sample in samples if sample.id in requested_ids]

    found_ids = {sample.id for sample in selected}

    missing_ids = requested_ids - found_ids

    if missing_ids:
        raise ValueError("Unknown sample IDs: " + ", ".join(sorted(missing_ids)))

    return selected


async def run_evaluation(
    dataset_path: str,
    output_path: str,
    ids: list[str] | None = None,
) -> None:

    samples = load_golden_dataset(dataset_path)

    samples = select_samples(
        samples,
        ids,
    )

    print(f"Loaded {len(samples)} " "evaluation samples.")

    graph, connection_pool = await get_compiled_graph()

    results: list[EvaluationRun] = []

    try:
        for sample in samples:
            result = await run_sample(
                sample,
                graph,
            )

            results.append(result)

            save_results(
                results,
                output_path,
            )
    finally:
        await connection_pool.close()

    successful = sum(result.status == "success" for result in results)

    failed = len(results) - successful

    print()
    print("=" * 70)
    print("Evaluation run complete")
    print("=" * 70)

    print(f"Successful: {successful}")

    print(f"Failed:     {failed}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=("Run golden dataset " "questions through CRAG.")
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    asyncio.run(
        run_evaluation(
            dataset_path=(args.dataset),
            output_path=(args.output),
            ids=args.ids,
        )
    )


if __name__ == "__main__":
    main()
