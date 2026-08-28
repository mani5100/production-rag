from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GoldenSample:
    id: str
    question: str
    expected_answer: str
    category: str
    source_doc: str | None = None
    source_page: int | None = None


@dataclass
class EvaluationRun:
    id: str
    question: str
    expected_answer: str
    category: str

    generated_answer: str = ""

    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)

    expected_source_doc: str | None = None
    expected_source_page: int | None = None

    standalone_query: str | None = None
    retrieval_queries: list[str] = field(default_factory=list)

    web_search_used: bool = False

    retrieval_attempts: int = 0
    generation_attempts: int = 0
    reflection_attempts: int = 0

    grade_verdict: str | None = None
    grade_reason: str | None = None

    reflection_action: str | None = None
    reflection_reason: str | None = None

    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)