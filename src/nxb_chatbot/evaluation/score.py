"""
Score a completed evaluation run (raw_runs.json) against RAGAS metrics.

Usage:
    uv run python -m nxb_chatbot.evaluation.score \
        --input evaluation_results/raw_runs.json \
        --output evaluation_results/scored_report.json
"""

import argparse
import json
import logging
from pathlib import Path

from nxb_chatbot.evaluation.metrics import (
    compute_faithfulness_batch,
    compute_ragas_metrics,
)
from nxb_chatbot.evaluation.models import EvaluationRun

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "evaluation_results/raw_runs.json"
DEFAULT_OUTPUT = "evaluation_results/scored_report.json"


def load_runs(path: str | Path) -> list[EvaluationRun]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No raw runs found at {path}. Run the evaluation harness "
            "(evaluation/runner.py) first."
        )

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return [EvaluationRun(**item) for item in raw]


def build_report(runs: list[EvaluationRun]) -> dict:
    successful_runs = [run for run in runs if run.status == "success"]
    skipped = len(runs) - len(successful_runs)

    if skipped:
        logger.warning("Skipping %d run(s) with status != 'success'", skipped)

    logger.info(
        "Scoring faithfulness (hand-rolled) for %d runs...", len(successful_runs)
    )
    faithfulness_results = compute_faithfulness_batch(successful_runs)

    scored_faithfulness = [r for r in faithfulness_results if r.score is not None]
    mean_faithfulness = (
        sum(r.score for r in scored_faithfulness) / len(scored_faithfulness)
        if scored_faithfulness
        else None
    )

    logger.info(
        "Scoring answer_relevancy / context_precision / context_recall via ragas..."
    )
    ragas_results = compute_ragas_metrics(successful_runs)

    faithfulness_by_id = {r.run_id: r for r in faithfulness_results}
    ragas_by_id = {row.get("user_input"): row for row in ragas_results["per_sample"]}

    per_sample = []
    for run in successful_runs:
        fr = faithfulness_by_id.get(run.id)
        rr = ragas_by_id.get(run.question, {})

        per_sample.append(
            {
                "id": run.id,
                "category": run.category,
                "question": run.question,
                "faithfulness": fr.score if fr else None,
                "num_claims": len(fr.claims) if fr else 0,
                "answer_relevancy": rr.get("answer_relevancy"),
                "context_precision": rr.get("context_precision"),
                "context_recall": rr.get("context_recall"),
            }
        )

    return {
        "summary": {
            "total_runs": len(runs),
            "scored_runs": len(successful_runs),
            "skipped_runs": skipped,
            "mean_faithfulness": mean_faithfulness,
            **ragas_results["mean"],
        },
        "per_sample": per_sample,
        "faithfulness_detail": [fr.to_dict() for fr in faithfulness_results],
    }


def save_report(report: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved scored report to %s", output_path)


def print_summary(summary: dict) -> None:
    print()
    print("=" * 70)
    print("RAGAS evaluation summary")
    print("=" * 70)
    print(f"Runs scored:        {summary['scored_runs']} / {summary['total_runs']}")
    print(f"Faithfulness:       {summary.get('mean_faithfulness')}")
    print(f"Answer relevancy:   {summary.get('answer_relevancy')}")
    print(f"Context precision:  {summary.get('context_precision')}")
    print(f"Context recall:     {summary.get('context_recall')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a completed evaluation run against RAGAS metrics."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = load_runs(args.input)
    report = build_report(runs)

    save_report(report, args.output)
    print_summary(report["summary"])


if __name__ == "__main__":
    main()
