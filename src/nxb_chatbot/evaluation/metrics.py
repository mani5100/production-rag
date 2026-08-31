"""
RAGAS-style evaluation metrics for CRAG evaluation runs.

Faithfulness is implemented by hand (claim decomposition + verification
against retrieved context, no ragas call). Answer relevancy, context
precision, and context recall are computed via the `ragas` library,
using the project's own Ollama LLM/embeddings so no extra API keys
are needed.
"""

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nxb_chatbot.core.embeddings import get_dense_embedder
from nxb_chatbot.evaluation.models import EvaluationRun
from nxb_chatbot.rag.services import llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-rolled metric: Faithfulness
# ---------------------------------------------------------------------------
#
# Faithfulness asks: "of everything the answer claims, how much is actually
# backed by the retrieved context?" It is NOT the same as correctness --
# an answer can be faithful to bad context and still be wrong overall
# (that's what answer relevancy / context recall catch instead).
#
# The standard RAGAS-style recipe, reproduced here manually:
#   1. Decompose the generated answer into a list of atomic, checkable
#      claims (one fact per claim, no compound sentences).
#   2. For each claim, ask the LLM a yes/no question: "can this claim be
#      directly inferred from the given context, using only the context
#      and not outside knowledge?"
#   3. score = (# claims verified as supported) / (# total claims)
#
# A score of 1.0 means every claim in the answer is grounded in the
# retrieved chunks. A score of 0.0 means the answer is entirely
# unsupported (i.e. hallucinated) by what was retrieved.

_CLAIM_DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template(
    """Break the given answer into a list of atomic, standalone factual
claims. Each claim must:
- express exactly one fact
- be understandable without reading the other claims (resolve any
  pronouns using the question for context)
- not include opinions, hedges, or meta-commentary about the answer itself

Return ONLY a JSON array of strings, nothing else. If the answer makes
no checkable factual claims (e.g. it's a refusal, greeting, or asks a
clarifying question), return an empty array: []

Question: {question}
Answer: {answer}

JSON array of claims:"""
)

_CLAIM_VERIFICATION_PROMPT = ChatPromptTemplate.from_template(
    """You are verifying whether a claim is supported by a given context.

Context:
{context}

Claim: {claim}

Answer strictly "yes" if the claim can be directly inferred from the
context alone. Answer strictly "no" if the claim requires information
not present in the context, or contradicts the context. Do not use
any outside knowledge. Respond with exactly one word: yes or no."""
)


@dataclass
class FaithfulnessResult:
    run_id: str
    score: float | None
    claims: list[str]
    verdicts: list[bool]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "faithfulness": self.score,
            "num_claims": len(self.claims),
            "num_supported": sum(self.verdicts),
            "claims": [
                {"claim": c, "supported": v} for c, v in zip(self.claims, self.verdicts)
            ],
        }


def _parse_claims(raw: str) -> list[str]:
    """Best-effort JSON-array parse; tolerates code fences / stray text."""
    raw = raw.strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse claim list, got: %r", raw)
        return []

    if not isinstance(claims, list):
        return []

    return [str(c).strip() for c in claims if str(c).strip()]


def decompose_into_claims(question: str, answer: str) -> list[str]:
    if not answer or not answer.strip():
        return []

    chain = _CLAIM_DECOMPOSITION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "answer": answer})
    return _parse_claims(raw)


def verify_claim(claim: str, context: str) -> bool:
    chain = _CLAIM_VERIFICATION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "claim": claim}).strip().lower()
    return raw.startswith("yes")


def compute_faithfulness(run: EvaluationRun) -> FaithfulnessResult:
    context = "\n\n---\n\n".join(run.retrieved_contexts)

    claims = decompose_into_claims(run.question, run.generated_answer)
    logger.info("[%s] decomposed answer into %d claim(s)", run.id, len(claims))

    if not claims:
        logger.info("[%s] no checkable claims -> score=None", run.id)
        return FaithfulnessResult(run_id=run.id, score=None, claims=[], verdicts=[])

    if not context.strip():
        logger.info("[%s] no retrieved context -> score=0.0", run.id)
        return FaithfulnessResult(
            run_id=run.id, score=0.0, claims=claims, verdicts=[False] * len(claims)
        )

    verdicts = []
    for i, claim in enumerate(claims, 1):
        verdict = verify_claim(claim, context)
        verdicts.append(verdict)
        logger.info(
            "[%s] claim %d/%d: %s -> %s",
            run.id, i, len(claims), claim[:80], "SUPPORTED" if verdict else "NOT SUPPORTED"
        )

    score = sum(verdicts) / len(verdicts)
    logger.info("[%s] faithfulness = %.2f", run.id, score)

    return FaithfulnessResult(run_id=run.id, score=score, claims=claims, verdicts=verdicts)


def compute_faithfulness_batch(runs: list[EvaluationRun]) -> list[FaithfulnessResult]:
    results = []
    for i, run in enumerate(runs, 1):
        logger.info("=== Faithfulness: sample %d/%d (id=%s) ===", i, len(runs), run.id)
        results.append(compute_faithfulness(run))
    return results


# ---------------------------------------------------------------------------
# Library metrics: answer relevancy, context precision, context recall
# ---------------------------------------------------------------------------
#
# These three are computed with `ragas` rather than by hand. They're
# wired to the project's own Ollama LLM + embedding model (via LangChain
# wrapper classes) so scoring doesn't require an OpenAI key.


def _build_ragas_dataset(runs: list[EvaluationRun]):
    from datasets import Dataset

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for run in runs:
        if run.status != "success":
            continue

        rows["question"].append(run.question)
        rows["answer"].append(run.generated_answer)
        rows["contexts"].append(run.retrieved_contexts or [""])
        rows["ground_truth"].append(run.expected_answer)

    return Dataset.from_dict(rows)


def _patch_ragas_vertexai_import() -> None:
    """
    ragas 0.4.3 unconditionally imports ChatVertexAI from
    langchain_community.chat_models.vertexai, a module dropped in
    langchain-community's "sunset" release (only langchain_community.llms.VertexAI
    remains). We never use Vertex AI here (Ollama only), so stub the missing
    submodule in sys.modules to let ragas import cleanly.
    """
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(module_name)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = stub


def compute_ragas_metrics(runs: list[EvaluationRun]) -> dict:
    """
    Computes answer_relevancy, context_precision, and context_recall
    over all successful runs using the ragas library, returning both
    per-sample scores and dataset-level averages.
    """
    _patch_ragas_vertexai_import()
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall

    dataset = _build_ragas_dataset(runs)

    if len(dataset) == 0:
        return {"per_sample": [], "mean": {}}

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(get_dense_embedder())

    result = evaluate(
        dataset,
        metrics=[answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    result_df = result.to_pandas()

    per_sample = result_df.to_dict(orient="records")

    mean = {
        "answer_relevancy": float(result_df["answer_relevancy"].mean()),
        "context_precision": float(result_df["context_precision"].mean()),
        "context_recall": float(result_df["context_recall"].mean()),
    }

    return {"per_sample": per_sample, "mean": mean}
