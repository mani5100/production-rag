import json
from pathlib import Path

from src.nxb_chatbot.evaluation.models import GoldenSample

def load_golden_dataset(path: str | Path) -> list[GoldenSample]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError(
            "Golden dataset must contain a JSON array."
        )

    samples: list[GoldenSample] = []

    for index, item in enumerate(raw_data):
        required_fields = {
            "id",
            "question",
            "expected_answer",
            "category",
        }

        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Sample at index {index} is missing fields: "
                f"{sorted(missing)}"
            )

        samples.append(
            GoldenSample(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                category=item["category"],
                source_doc=item.get("source_doc"),
                source_page=item.get("source_page"),
            )
        )

    return samples